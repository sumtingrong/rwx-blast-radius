"""Blast Radius selector: dynamic line/fixture/dialect/component coverage + static AST/attr analysis + 0..8 worker packing."""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

DATA = Path(os.environ.get("BLAST_DATA", "/data"))
REF = Path(os.environ.get("SQLGLOT_REF", "/app/sqlglot_ref"))
_DEFAULT_MAPS = "/app/maps.pkl" if Path("/app/maps.pkl").exists() else "/app/maps.json.gz"
MAPS_PATH = Path(os.environ.get("BLAST_MAPS", _DEFAULT_MAPS))

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
BASE_CLASSES = {"Dialect", "Parser", "BaseParser", "Generator", "Tokenizer", "JSONPathTokenizer", "_TokenizerBase"}
ONCE_FUNCS = {"__new__", "__init_subclass__", "_build_dispatch"}
GENERIC_EXPR_ATTRS = {"arg_types", "is_var_len_args", "_sql_names", "__slots__"}


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

        if k > 1 and len(ordered) <= 400:
            for _ in range(40):
                hi = loads.index(max(loads))
                cur_max = loads[hi]
                best_action = None
                best_new_max = cur_max

                # Try moving 1 test from hi -> any lighter worker j
                for idx_t, t in enumerate(workers[hi]):
                    dt = dur_ms[t]
                    new_hi = loads[hi] - dt
                    for j in range(k):
                        if j == hi:
                            continue
                        new_j = loads[j] + dt
                        cand_max = new_hi if new_hi > new_j else new_j
                        if cand_max < best_new_max:
                            best_new_max = cand_max
                            best_action = ("move", idx_t, j)

                # If no improving single move, try 1-1 swap between hi and j
                if best_action is None and len(ordered) <= 200:
                    for idx_a, ta in enumerate(workers[hi]):
                        da = dur_ms[ta]
                        for j in range(k):
                            if j == hi:
                                continue
                            for idx_b, tb in enumerate(workers[j]):
                                db = dur_ms[tb]
                                if da <= db:
                                    continue
                                new_hi = loads[hi] - da + db
                                new_j = loads[j] - db + da
                                cand_max = new_hi if new_hi > new_j else new_j
                                if cand_max < best_new_max:
                                    best_new_max = cand_max
                                    best_action = ("swap", idx_a, j, idx_b)

                if best_action is None:
                    break
                if best_action[0] == "move":
                    _, idx_t, j = best_action
                    t = workers[hi].pop(idx_t)
                    workers[j].append(t)
                    loads[hi] -= dur_ms[t]
                    loads[j] += dur_ms[t]
                else:
                    _, idx_a, j, idx_b = best_action
                    ta = workers[hi][idx_a]
                    tb = workers[j][idx_b]
                    workers[hi][idx_a] = tb
                    workers[j][idx_b] = ta
                    diff = dur_ms[ta] - dur_ms[tb]
                    loads[hi] -= diff
                    loads[j] += diff

        cost = max(loads) + 2000 * k
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_workers = workers

    return [w for w in best_workers if w]


def _load_maps(path: Path) -> dict:
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    tests_raw = json.loads((DATA / "tests.json").read_text("utf-8"))["tests"]
    id_to_num = {t["id"]: t["number"] for t in tests_raw}
    dur_ms = {t["number"]: round(t["seconds"] * 1000) for t in tests_raw}
    tests_by_file: dict[str, list[int]] = defaultdict(list)
    for t in tests_raw:
        tests_by_file[t["file"]].append(t["number"])
    all_test_nums = {t["number"] for t in tests_raw}

    maps = _load_maps(MAPS_PATH)

    idx_to_num: dict[int, int] = {}
    for idx, nodeid in enumerate(maps["nodeids"]):
        if nodeid in id_to_num:
            idx_to_num[idx] = id_to_num[nodeid]

    def remap_dict(raw: dict) -> dict[str, set[int]]:
        out = {}
        for k, v in raw.items():
            s = {idx_to_num[i] for i in v if i in idx_to_num}
            if s:
                out[k] = s
        return out

    line_to_tests = remap_dict(maps["line_to_idxs"])
    fixture_to_tests = remap_dict(maps["fixture_to_idxs"])
    dialect_to_tests = remap_dict(maps["dialect_to_idxs"])
    tok_to_tests = remap_dict(maps.get("tok_to_idxs", {}))
    parse_to_tests = remap_dict(maps.get("parse_to_idxs", {}))
    gen_init_to_tests = remap_dict(maps.get("gen_init_to_idxs", {}))
    gen_sql_to_tests = remap_dict(maps.get("gen_sql_to_idxs", {}))

    dialect_subclasses = maps["dialect_subclasses"]
    parser_subclasses = maps["parser_subclasses"]
    generator_subclasses = maps["generator_subclasses"]
    tokenizer_subclasses = maps.get("tokenizer_subclasses", {})
    file_data = maps["file_data"]
    attr_readers = maps["attr_readers"]
    name_readers = maps["name_readers"]
    mod_attr_readers = maps.get("mod_attr_readers", {})
    mod_name_readers = maps.get("mod_name_readers", {})
    file_deps = maps.get("file_deps", {})

    # reverse_importers[target_rel] = test numbers of all test files that transitively import target_rel
    file_importers: dict[str, set[int]] = defaultdict(set)
    mod_importers: dict[str, set[str]] = defaultdict(set)
    for rel_src, deps in file_deps.items():
        if rel_src.startswith("tests/") and rel_src in tests_by_file:
            tnums = tests_by_file[rel_src]
            file_importers[rel_src].update(tnums)
            for d in deps:
                file_importers[d].update(tnums)
        for d in deps:
            mod_importers[d].add(rel_src)
        mod_importers[rel_src].add(rel_src)

    file_cov_tests: dict[str, set[int]] = defaultdict(set)
    for key, tset in line_to_tests.items():
        rel, _ = key.rsplit(":", 1)
        file_cov_tests[rel].update(tset)

    orig_src_cache: dict[str, str] = {}

    def get_orig(rel: str) -> str:
        if rel not in orig_src_cache:
            p = REF / rel
            orig_src_cache[rel] = p.read_text("utf-8") if p.exists() else ""
        return orig_src_cache[rel]

    def get_line_info(fdata: dict, ln: int) -> dict:
        li = fdata["line_info"]
        return li.get(ln) or li.get(str(ln), {})

    def get_stmt_span(fdata: dict, ln: int) -> list[int] | None:
        ss = fdata["stmt_spans"]
        return ss.get(ln) or ss.get(str(ln))

    def tests_for_line_or_span(rel: str, ln: int) -> set[int]:
        out = set(line_to_tests.get(f"{rel}:{ln}", ()))
        fdata = file_data.get(rel)
        if fdata:
            span = get_stmt_span(fdata, ln)
            if span:
                for sln in range(span[0], span[1] + 1):
                    out.update(line_to_tests.get(f"{rel}:{sln}", ()))
        return out

    def classify_class(cls_name: str) -> tuple[str, list[str]] | tuple[None, None]:
        if cls_name in BASE_CLASSES or cls_name.split(".")[-1] in ("_TokenizerBase",):
            return None, None
        if cls_name in dialect_subclasses:
            return "dialect", sorted(dialect_subclasses[cls_name])
        if cls_name in parser_subclasses:
            return "parser", sorted(parser_subclasses[cls_name])
        if cls_name in generator_subclasses:
            return "generator", sorted(generator_subclasses[cls_name])
        if cls_name in tokenizer_subclasses:
            return "tokenizer", sorted(tokenizer_subclasses[cls_name])
        return None, None

    def component_tests_for_reader(d_names: list[str], r_rel: str, r_ln: int) -> set[int]:
        if r_rel in ("sqlglot/tokens.py", "sqlglot/tokenizer_core.py"):
            source = tok_to_tests or dialect_to_tests
        elif r_rel == "sqlglot/parser.py" or r_rel.startswith("sqlglot/parsers/"):
            source = parse_to_tests or dialect_to_tests
        elif r_rel == "sqlglot/generator.py" or r_rel.startswith("sqlglot/generators/"):
            r_info = get_line_info(file_data[r_rel], r_ln) if r_rel in file_data else {}
            if r_info.get("func") == "__init__":
                source = gen_init_to_tests or dialect_to_tests
            else:
                source = gen_sql_to_tests or dialect_to_tests
        else:
            source = dialect_to_tests
        res: set[int] = set()
        for dn in d_names:
            res.update(source.get(dn, ()))
        return res

    def union_dialect_tests(d_names: list[str]) -> set[int]:
        res: set[int] = set()
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
                if not rel.endswith(".py") or fdiff["new_file"] or fdiff["deleted_file"]:
                    selected.update(file_importers.get(rel, all_test_nums))
                    continue

                fdata = file_data.get(rel)
                orig_src = get_orig(rel)
                patched_src = apply_hunks(orig_src, fdiff["hunks"]) if orig_src else None
                if patched_src is not None and fdata:
                    try:
                        new_ast = ast.parse(patched_src)
                    except SyntaxError:
                        selected.update(file_importers.get(rel, all_test_nums))
                        continue
                    new_hash = hashlib.blake2b(ast.dump(new_ast).encode("utf-8"), digest_size=16).hexdigest()
                    if new_hash == fdata["ast_hash"]:
                        continue

                # If rel is a test file and no other test file imports it for non-test code, try method-level granularity
                if rel in tests_by_file and fdata and fdata.get("test_methods"):
                    test_methods = fdata["test_methods"]
                    matched_tests: set[int] = set()
                    all_inside_test_methods = True
                    for h in fdiff["hunks"]:
                        old_ln = h["old_start"]
                        touched: list[int] = []
                        inserts: list[int] = []
                        for hl in h["lines"]:
                            if not hl:
                                continue
                            tag = hl[0]
                            if tag == " ":
                                old_ln += 1
                            elif tag == "-":
                                touched.append(old_ln)
                                old_ln += 1
                            elif tag == "+":
                                inserts.append(old_ln)
                        for ln in touched:
                            hit = None
                            for cls_n, meth_n, s_ln, e_ln in test_methods:
                                if s_ln <= ln <= e_ln:
                                    hit = f"{rel}::{cls_n}::{meth_n}"
                                    break
                            if hit and hit in id_to_num:
                                matched_tests.add(id_to_num[hit])
                            else:
                                all_inside_test_methods = False
                                break
                        if not all_inside_test_methods:
                            break
                        if not touched:
                            for ln in inserts:
                                hit = None
                                for cls_n, meth_n, s_ln, e_ln in test_methods:
                                    if s_ln < ln <= e_ln:
                                        hit = f"{rel}::{cls_n}::{meth_n}"
                                        break
                                if hit and hit in id_to_num:
                                    matched_tests.add(id_to_num[hit])
                                else:
                                    all_inside_test_methods = False
                                    break
                        if not all_inside_test_methods:
                            break
                    if all_inside_test_methods and matched_tests:
                        selected.update(matched_tests)
                        continue

                selected.update(file_importers.get(rel, all_test_nums))
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

            fdata = file_data.get(rel)
            orig_src = get_orig(rel)
            patched_src = apply_hunks(orig_src, fdiff["hunks"]) if orig_src else None
            if patched_src is not None and fdata:
                try:
                    new_ast = ast.parse(patched_src)
                except SyntaxError:
                    selected.update(file_importers.get(rel, all_test_nums))
                    continue
                new_hash = hashlib.blake2b(ast.dump(new_ast).encode("utf-8"), digest_size=16).hexdigest()
                if new_hash == fdata["ast_hash"]:
                    continue

            if rel.startswith("sqlglot/typing/"):
                selected.update(tests_by_file.get("tests/test_typing.py", ()))

            if not fdata:
                selected.update(all_test_nums)
                continue

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
                    info = get_line_info(fdata, ln)
                    if info.get("is_lambda"):
                        lr = info["lambda_range"]
                        for l2 in range(lr[0], lr[1] + 1):
                            selected.update(tests_for_line_or_span(rel, l2))
                    elif info.get("func"):
                        if info["func"] in ONCE_FUNCS:
                            selected.update(file_importers.get(rel, all_test_nums))
                            continue
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
                        is_expr_class_attr = rel.startswith("sqlglot/expressions/") and attr_name in GENERIC_EXPR_ATTRS
                        reader_locs: list[str] = []
                        if attr_name and not is_expr_class_attr:
                            reader_locs.extend(attr_readers.get(attr_name, ()))
                        if cls_name:
                            short_cls = cls_name.split(".")[-1]
                            reader_locs.extend(name_readers.get(cls_name, ()))
                            reader_locs.extend(attr_readers.get(cls_name, ()))
                            if short_cls != cls_name:
                                reader_locs.extend(name_readers.get(short_cls, ()))
                                reader_locs.extend(attr_readers.get(short_cls, ()))

                        importers_of_rel = mod_importers.get(rel, {rel})
                        cls_kind, d_names = classify_class(cls_name) if cls_name else (None, None)
                        if d_names is not None:
                            if reader_locs:
                                matched = set()
                                for loc in reader_locs:
                                    r_rel, r_ln_str = loc.rsplit(":", 1)
                                    r_ln = int(r_ln_str)
                                    t_here = tests_for_line_or_span(r_rel, r_ln)
                                    if t_here:
                                        matched.update(t_here & component_tests_for_reader(d_names, r_rel, r_ln))
                                selected.update(matched)
                            else:
                                selected.update(union_dialect_tests(d_names))
                            if attr_name:
                                for mrel in mod_attr_readers.get(attr_name, ()):
                                    if mrel in importers_of_rel:
                                        selected.update(file_importers.get(mrel, ()))
                        elif reader_locs:
                            for loc in reader_locs:
                                r_rel, r_ln_str = loc.rsplit(":", 1)
                                selected.update(tests_for_line_or_span(r_rel, int(r_ln_str)))
                            if not is_expr_class_attr and attr_name:
                                for mrel in mod_attr_readers.get(attr_name, ()):
                                    if mrel in importers_of_rel:
                                        selected.update(file_importers.get(mrel, ()))
                            if cls_name:
                                for mrel in mod_attr_readers.get(cls_name, ()):
                                    if mrel in importers_of_rel:
                                        selected.update(file_importers.get(mrel, ()))
                        elif not is_expr_class_attr:
                            selected.update(file_cov_tests.get(rel, ()))
                    elif "module_var" in info:
                        var_name = info["module_var"]
                        importers_of_rel = mod_importers.get(rel, {rel})
                        reader_locs = []
                        if var_name:
                            for loc in name_readers.get(var_name, ()):
                                r_rel, _ = loc.rsplit(":", 1)
                                if r_rel in importers_of_rel:
                                    reader_locs.append(loc)
                            for loc in attr_readers.get(var_name, ()):
                                r_rel, _ = loc.rsplit(":", 1)
                                if r_rel in importers_of_rel:
                                    reader_locs.append(loc)
                            for mrel in mod_name_readers.get(var_name, ()):
                                if mrel in importers_of_rel:
                                    selected.update(file_importers.get(mrel, ()))
                            for mrel in mod_attr_readers.get(var_name, ()):
                                if mrel in importers_of_rel:
                                    selected.update(file_importers.get(mrel, ()))
                        if reader_locs:
                            for loc in reader_locs:
                                r_rel, r_ln_str = loc.rsplit(":", 1)
                                selected.update(tests_for_line_or_span(r_rel, int(r_ln_str)))
                        elif not (var_name and (mod_name_readers.get(var_name) or mod_attr_readers.get(var_name))):
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
