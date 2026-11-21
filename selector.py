"""Blast Radius selector: dynamic line/fixture/dialect coverage + static AST/attr analysis + 0..8 worker packing."""
from __future__ import annotations

import ast
import gzip
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

DATA = Path(os.environ.get("BLAST_DATA", "/data"))
REF = Path(os.environ.get("SQLGLOT_REF", "/app/sqlglot_ref"))
MAPS_PATH = Path(os.environ.get("BLAST_MAPS", "/app/maps.json.gz"))

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
BASE_CLASSES = {"Dialect", "Parser", "BaseParser", "Generator", "Tokenizer"}


def parse_patch(patch_text: str) -> list[dict]:
    files = []
    cur = None
    lines = patch_text.splitlines(keepends=True)
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("diff --git "):
            if cur is not None:
                files.append(cur)
            cur = {
                "old_path": None,
                "new_path": None,
                "hunks": [],
                "new_file": False,
                "deleted_file": False,
            }
            i += 1
            continue
        if cur is None:
            i += 1
            continue
        if line.startswith("new file mode"):
            cur["new_file"] = True
        elif line.startswith("deleted file mode"):
            cur["deleted_file"] = True
        elif line.startswith("--- "):
            p = line[4:].strip()
            cur["old_path"] = None if p == "/dev/null" else (p[2:] if p.startswith("a/") else p)
        elif line.startswith("+++ "):
            p = line[4:].strip()
            cur["new_path"] = None if p == "/dev/null" else (p[2:] if p.startswith("b/") else p)
        elif line.startswith("@@ "):
            m = HUNK_RE.match(line)
            old_start = int(m.group(1))
            old_count = int(m.group(2)) if m.group(2) is not None else 1
            i += 1
            hunk_lines = []
            while i < n and not lines[i].startswith(("diff --git ", "@@ ")):
                if lines[i].startswith("\\ No newline"):
                    i += 1
                    continue
                hunk_lines.append(lines[i])
                i += 1
            cur["hunks"].append({
                "old_start": old_start,
                "old_count": old_count,
                "lines": hunk_lines,
            })
            continue
        i += 1
    if cur is not None:
        files.append(cur)
    return files


def apply_hunks(orig_text: str, hunks: list[dict]) -> str | None:
    orig_lines = orig_text.splitlines(keepends=True)
    out = []
    pos = 0
    for h in hunks:
        start_idx = max(0, h["old_start"] - 1) if h["old_count"] > 0 else h["old_start"]
        if start_idx < pos or start_idx > len(orig_lines):
            return None
        out.extend(orig_lines[pos:start_idx])
        pos = start_idx
        for hl in h["lines"]:
            if not hl:
                continue
            tag = hl[0]
            body = hl[1:]
            if tag == " ":
                out.append(orig_lines[pos] if pos < len(orig_lines) else body)
                pos += 1
            elif tag == "-":
                pos += 1
            elif tag == "+":
                out.append(body)
    out.extend(orig_lines[pos:])
    return "".join(out)


def pack_workers(test_nums: set[int], dur_ms: dict[int, int]) -> list[list[int]]:
    if not test_nums:
        return []
    ordered = sorted(test_nums, key=lambda t: (-dur_ms[t], t))
    best_cost = None
    best_workers = None
    max_k = min(8, len(ordered))

    for k in range(1, max_k + 1):
        workers = [[] for _ in range(k)]
        loads = [0] * k
        for t in ordered:
            idx = loads.index(min(loads))
            workers[idx].append(t)
            loads[idx] += dur_ms[t]

        # Fast local move refinement on busiest worker
        if k > 1 and len(ordered) <= 300:
            for _ in range(20):
                hi = loads.index(max(loads))
                lo = loads.index(min(loads))
                if hi == lo:
                    break
                cur_max = loads[hi]
                best_move = None
                best_new_max = cur_max
                for idx_t, t in enumerate(workers[hi]):
                    d = dur_ms[t]
                    new_hi = loads[hi] - d
                    new_lo = loads[lo] + d
                    cand_max = max(new_hi, new_lo)
                    if cand_max < best_new_max:
                        best_new_max = cand_max
                        best_move = idx_t
                if best_move is None:
                    break
                t = workers[hi].pop(best_move)
                workers[lo].append(t)
                loads[hi] -= dur_ms[t]
                loads[lo] += dur_ms[t]

        cost = max(loads) + 2000 * k
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_workers = workers

    return [w for w in best_workers if w]


def main() -> None:
    tests_raw = json.loads((DATA / "tests.json").read_text("utf-8"))["tests"]
    id_to_num = {t["id"]: t["number"] for t in tests_raw}
    dur_ms = {t["number"]: round(t["seconds"] * 1000) for t in tests_raw}
    tests_by_file: dict[str, list[int]] = defaultdict(list)
    for t in tests_raw:
        tests_by_file[t["file"]].append(t["number"])
    all_test_nums = {t["number"] for t in tests_raw}

    with gzip.open(MAPS_PATH, "rt", encoding="utf-8") as f:
        maps = json.load(f)

    # Remap map indices -> test numbers in /data/tests.json
    idx_to_num: dict[int, int] = {}
    for idx, nodeid in enumerate(maps["nodeids"]):
        if nodeid in id_to_num:
            idx_to_num[idx] = id_to_num[nodeid]

    def remap(idxs: list[int]) -> set[int]:
        return {idx_to_num[i] for i in idxs if i in idx_to_num}

    line_to_tests: dict[str, set[int]] = {
        k: remap(v) for k, v in maps["line_to_idxs"].items()
    }
    fixture_to_tests: dict[str, set[int]] = {
        k: remap(v) for k, v in maps["fixture_to_idxs"].items()
    }
    dialect_to_tests: dict[str, set[int]] = {
        k: remap(v) for k, v in maps["dialect_to_idxs"].items()
    }

    dialect_subclasses = maps["dialect_subclasses"]
    parser_subclasses = maps["parser_subclasses"]
    generator_subclasses = maps["generator_subclasses"]
    file_data = maps["file_data"]
    attr_readers = maps["attr_readers"]
    name_readers = maps["name_readers"]
    test_file_imports = maps["test_file_imports"]

    file_importers: dict[str, set[int]] = defaultdict(set)
    for tf, mods in test_file_imports.items():
        tnums = tests_by_file.get(tf, ())
        for m in mods:
            file_importers[m].update(tnums)

    file_cov_tests: dict[str, set[int]] = defaultdict(set)
    for key, tset in line_to_tests.items():
        rel, _ = key.rsplit(":", 1)
        file_cov_tests[rel].update(tset)

    orig_src_cache: dict[str, str] = {}
    orig_dump_cache: dict[str, str] = {}

    def get_orig(rel: str) -> str:
        if rel not in orig_src_cache:
            p = REF / rel
            orig_src_cache[rel] = p.read_text("utf-8") if p.exists() else ""
        return orig_src_cache[rel]

    def get_orig_dump(rel: str) -> str:
        if rel not in orig_dump_cache:
            orig_dump_cache[rel] = ast.dump(ast.parse(get_orig(rel)))
        return orig_dump_cache[rel]

    def tests_for_line_or_span(rel: str, ln: int) -> set[int]:
        out = set(line_to_tests.get(f"{rel}:{ln}", ()))
        fdata = file_data.get(rel)
        if fdata:
            span = fdata["stmt_spans"].get(str(ln))
            if span:
                for sln in range(span[0], span[1] + 1):
                    out.update(line_to_tests.get(f"{rel}:{sln}", ()))
        return out

    def dialect_tests_for_class(cls_name: str, rel: str) -> set[int] | None:
        if cls_name in BASE_CLASSES:
            return None
        d_names = None
        if cls_name in dialect_subclasses:
            d_names = dialect_subclasses[cls_name]
        elif cls_name in parser_subclasses:
            d_names = parser_subclasses[cls_name]
        elif cls_name in generator_subclasses:
            d_names = generator_subclasses[cls_name]
        if not d_names:
            return None
        res = set(file_cov_tests.get(rel, ()))
        for dn in d_names:
            res.update(dialect_to_tests.get(dn, ()))
        return res

    def select_for_patch(patch_text: str) -> set[int]:
        selected: set[int] = set()
        for fdiff in parse_patch(patch_text):
            rel = fdiff["new_path"] or fdiff["old_path"]
            if not rel:
                continue

            # 1. Fixture files
            if rel.startswith("tests/fixtures/"):
                if rel.endswith(".sql"):
                    non_comment = False
                    for h in fdiff["hunks"]:
                        for hl in h["lines"]:
                            if hl and hl[0] in ("+", "-"):
                                body = hl[1:].strip()
                                if body and not body.startswith("--"):
                                    non_comment = True
                                    break
                        if non_comment:
                            break
                    if not non_comment:
                        continue

                if rel in fixture_to_tests and fixture_to_tests[rel]:
                    selected.update(fixture_to_tests[rel])
                else:
                    if "tpc-h" in rel or "tpc-ds" in rel:
                        selected.update(tests_by_file["tests/test_executor.py"])
                    elif "optimizer/" in rel:
                        selected.update(tests_by_file["tests/test_optimizer.py"])
                    else:
                        selected.update(tests_by_file["tests/test_transpile.py"])
                        selected.update(tests_by_file["tests/test_serde.py"])
                continue

            # 2. Test files
            if rel.startswith("tests/"):
                if rel in tests_by_file:
                    selected.update(tests_by_file[rel])
                else:
                    selected.update(all_test_nums)
                continue

            # 3. Non-sqlglot files
            if not rel.startswith("sqlglot/"):
                if "sqlglotc" in rel:
                    selected.update(tests_by_file.get("tests/test_sqlglotc.py", ()))
                continue

            if not rel.endswith(".py"):
                continue

            if fdiff["new_file"] or fdiff["deleted_file"]:
                selected.update(file_importers.get(rel, all_test_nums))
                continue

            orig_src = get_orig(rel)
            patched_src = apply_hunks(orig_src, fdiff["hunks"])
            if patched_src is not None:
                try:
                    new_ast = ast.parse(patched_src)
                except SyntaxError:
                    selected.update(file_importers.get(rel, all_test_nums))
                    continue
                if ast.dump(new_ast) == get_orig_dump(rel):
                    continue

            if rel.startswith("sqlglot/typing/"):
                selected.update(tests_by_file.get("tests/test_typing.py", ()))

            fdata = file_data.get(rel)
            if not fdata:
                selected.update(all_test_nums)
                continue

            line_info = fdata["line_info"]
            for h in fdiff["hunks"]:
                old_ln = h["old_start"]
                touched_old_lines: list[int] = []
                insertion_anchors: list[int] = []
                for hl in h["lines"]:
                    if not hl:
                        continue
                    tag = hl[0]
                    if tag == " ":
                        old_ln += 1
                    elif tag == "-":
                        touched_old_lines.append(old_ln)
                        old_ln += 1
                    elif tag == "+":
                        insertion_anchors.append(old_ln - 1)
                        insertion_anchors.append(old_ln)

                candidate_lines = touched_old_lines if touched_old_lines else insertion_anchors
                for ln in candidate_lines:
                    if ln < 1:
                        continue
                    info = line_info.get(str(ln), {})
                    if info.get("is_lambda"):
                        lr = info["lambda_range"]
                        for l2 in range(lr[0], lr[1] + 1):
                            selected.update(tests_for_line_or_span(rel, l2))
                    elif info.get("func"):
                        cov_here = tests_for_line_or_span(rel, ln)
                        if cov_here:
                            selected.update(cov_here)
                        else:
                            fr = info["func_range"]
                            if ln <= fr[0] + 1:
                                for l2 in range(fr[0], fr[1] + 1):
                                    selected.update(line_to_tests.get(f"{rel}:{l2}", ()))
                            elif not touched_old_lines:
                                for delta in (-1, 1, -2, 2):
                                    selected.update(tests_for_line_or_span(rel, ln + delta))
                    elif "class_attr" in info:
                        cls_name, attr_name = info["class_attr"]
                        reader_locs = []
                        is_expr_class_attr = rel.startswith("sqlglot/expressions/") and attr_name in {
                            "arg_types",
                            "is_var_len_args",
                            "_sql_names",
                            "__slots__",
                        }
                        if attr_name and not is_expr_class_attr:
                            reader_locs.extend(attr_readers.get(attr_name, ()))
                        if cls_name:
                            reader_locs.extend(name_readers.get(cls_name, ()))

                        reader_tests: set[int] = set()
                        for loc in reader_locs:
                            r_rel, r_ln = loc.rsplit(":", 1)
                            reader_tests.update(tests_for_line_or_span(r_rel, int(r_ln)))

                        d_tests = dialect_tests_for_class(cls_name, rel) if cls_name else None
                        if d_tests is not None:
                            if reader_tests:
                                selected.update(reader_tests & d_tests)
                            else:
                                selected.update(d_tests)
                        elif reader_locs:
                            selected.update(reader_tests)
                        elif not is_expr_class_attr:
                            selected.update(file_cov_tests.get(rel, ()))
                    elif "module_var" in info:
                        var_name = info["module_var"]
                        reader_locs = name_readers.get(var_name, ()) if var_name else ()
                        if reader_locs:
                            for loc in reader_locs:
                                r_rel, r_ln = loc.rsplit(":", 1)
                                selected.update(tests_for_line_or_span(r_rel, int(r_ln)))
                        else:
                            selected.update(file_cov_tests.get(rel, ()))
                    else:
                        selected.update(file_cov_tests.get(rel, ()))

        return selected

    for patch in sorted((DATA / "commits").glob("*.patch")):
        sel = select_for_patch(patch.read_text("utf-8"))
        workers = pack_workers(sel, dur_ms)
        sys.stdout.write(json.dumps({"commit": patch.stem, "workers": workers}) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
