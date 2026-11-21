"""Blast Radius selector: lazy split-pickle loading + dynamic/AST/component coverage + 0..8 worker packing."""
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
_RUNTIME_ANN_FILES = {"sqlglot/executor/env.py"}
_CLASS_ANN_FILES = {
    "sqlglot/executor/env.py",
    "sqlglot/_typing.py",
    "sqlglot/diff.py",
    "sqlglot/lineage.py",
    "sqlglot/optimizer/pushdown_projections.py",
}
REVERSIBLE_CMPS = {
    ("Is", "Eq"),
    ("Eq", "Is"),
    ("IsNot", "NotEq"),
    ("NotEq", "IsNot"),
    ("Lt", "LtE"),
    ("LtE", "Lt"),
    ("Gt", "GtE"),
    ("GtE", "Gt"),
}
NEG_INVERT_OPS = {
    "Is": "IsNot",
    "IsNot": "Is",
    "In": "NotIn",
    "NotIn": "In",
}


def is_comment_or_blank_only(hunks: list[dict], str_lines) -> bool:
    for h in hunks:
        old_ln = h["old_start"]
        for hl in h["lines"]:
            if not hl:
                continue
            tag = hl[0]
            if tag == " ":
                old_ln += 1
            elif tag == "-":
                body = hl[1:].lstrip()
                if body and not body.startswith("#"):
                    return False
                if old_ln in str_lines:
                    return False
                old_ln += 1
            elif tag == "+":
                body = hl[1:].lstrip()
                if body and not body.startswith("#"):
                    return False
                if old_ln in str_lines or (old_ln - 1) in str_lines:
                    return False
    return True


def ast_equal(a, b) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, ast.AST):
        for f in a._fields:
            if not ast_equal(getattr(a, f), getattr(b, f)):
                return False
        return True
    if isinstance(a, list):
        return len(a) == len(b) and all(ast_equal(x, y) for x, y in zip(a, b))
    return a == b


def _is_const_none_or_bool(n: ast.AST) -> bool:
    return isinstance(n, ast.Constant) and (n.value is None or isinstance(n.value, bool))


def _is_const_none(n: ast.AST) -> bool:
    return isinstance(n, ast.Constant) and n.value is None


def _can_be_bool_none(n: ast.AST) -> bool:
    return (
        not isinstance(n, (ast.Compare, ast.Constant, ast.BoolOp))
        and not (isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not))
    )


def _is_type_checking(test: ast.AST) -> bool:
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and isinstance(test.value, ast.Name) and test.attr == "TYPE_CHECKING":
        return True
    return False


def _filter_tc_stmts(stmts: list[ast.stmt]) -> list[ast.stmt]:
    out = []
    for s in stmts:
        if isinstance(s, ast.If) and _is_type_checking(s.test):
            if s.orelse:
                out.extend(_filter_tc_stmts(s.orelse))
        else:
            out.append(s)
    return out


def has_future_annotations(tree: ast.AST) -> bool:
    if not isinstance(tree, ast.Module):
        return False
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            if any(al.name == "annotations" for al in node.names):
                return True
        elif not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            break
    return False


def _static_ast_key_tag(k: ast.AST) -> str | None:
    if isinstance(k, ast.Constant):
        if isinstance(k.value, (str, int, float, bool)) or k.value is None:
            return f"s:{k.value!r}"
        return None
    if isinstance(k, ast.Attribute):
        if isinstance(k.value, ast.Name):
            if k.value.id == "exp":
                return f"t:{k.attr}"
            if k.value.id == "TokenType":
                return f"e:TokenType.{k.attr}"
            if k.value.id == "DType":
                return f"e:DType.{k.attr}"
        elif isinstance(k.value, ast.Attribute):
            if k.value.attr == "TokenType":
                return f"e:TokenType.{k.attr}"
            if k.value.attr in ("DType", "Type"):
                return f"e:DType.{k.attr}"
    return None


def _diff_dict_nodes(old_d: ast.Dict, new_d: ast.Dict):
    old_spreads = []
    old_map = {}
    old_untracked = {}
    for k, v in zip(old_d.keys, old_d.values):
        if k is None:
            old_spreads.append(v)
        else:
            kt = _static_ast_key_tag(k)
            if kt is not None:
                if kt in old_map:
                    return None
                old_map[kt] = v
            else:
                uk = ast.dump(k)
                if uk in old_untracked:
                    return None
                old_untracked[uk] = v

    new_spreads = []
    new_map = {}
    new_untracked = {}
    for k, v in zip(new_d.keys, new_d.values):
        if k is None:
            new_spreads.append(v)
        else:
            kt = _static_ast_key_tag(k)
            if kt is not None:
                if kt in new_map:
                    return None
                new_map[kt] = v
            else:
                uk = ast.dump(k)
                if uk in new_untracked:
                    return None
                new_untracked[uk] = v

    if len(old_spreads) != len(new_spreads) or not all(ast_equal(ov, nv) for ov, nv in zip(old_spreads, new_spreads)):
        return None
    if set(old_map.keys()) != set(new_map.keys()):
        return None
    if set(old_untracked.keys()) != set(new_untracked.keys()):
        return None

    lambda_pairs = []
    for uk, ov in old_untracked.items():
        nv = new_untracked[uk]
        if not ast_equal(ov, nv):
            if isinstance(ov, ast.Lambda) and isinstance(nv, ast.Lambda) and ast_equal(ov.args, nv.args):
                lambda_pairs.append((ov.body, nv.body))
            else:
                return None

    changed_keys = set()
    for k, ov in old_map.items():
        nv = new_map[k]
        if not ast_equal(ov, nv):
            if isinstance(ov, ast.Lambda) and isinstance(nv, ast.Lambda) and ast_equal(ov.args, nv.args):
                lambda_pairs.append((ov.body, nv.body))
            else:
                changed_keys.add(k)
    return changed_keys, lambda_pairs


def _unwrap_not_cmp_invert(o: ast.AST, n: ast.AST) -> bool:
    if (
        isinstance(o, ast.UnaryOp)
        and isinstance(o.op, ast.Not)
        and isinstance(o.operand, ast.Compare)
        and len(o.operand.ops) == 1
        and len(o.operand.comparators) == 1
        and isinstance(n, ast.Compare)
        and len(n.ops) == 1
        and len(n.comparators) == 1
    ):
        op_o = type(o.operand.ops[0]).__name__
        op_n = type(n.ops[0]).__name__
        if NEG_INVERT_OPS.get(op_o) == op_n:
            return ast_equal(o.operand.left, n.left) and ast_equal(o.operand.comparators[0], n.comparators[0])
    return False


def _pure_neg_equiv(a: ast.AST, b: ast.AST) -> bool:
    if isinstance(a, ast.UnaryOp) and isinstance(a.op, ast.Not) and ast_equal(a.operand, b):
        return True
    if isinstance(b, ast.UnaryOp) and isinstance(b.op, ast.Not) and ast_equal(b.operand, a):
        return True
    if _unwrap_not_cmp_invert(a, b) or _unwrap_not_cmp_invert(b, a):
        return True
    if (
        isinstance(a, ast.Compare)
        and len(a.ops) == 1
        and len(a.comparators) == 1
        and isinstance(b, ast.Compare)
        and len(b.ops) == 1
        and len(b.comparators) == 1
    ):
        if NEG_INVERT_OPS.get(type(a.ops[0]).__name__) == type(b.ops[0]).__name__:
            if ast_equal(a.left, b.left) and ast_equal(a.comparators[0], b.comparators[0]):
                return True
    if (
        isinstance(a, ast.BoolOp)
        and isinstance(b, ast.BoolOp)
        and type(a.op) is not type(b.op)
        and len(a.values) == len(b.values)
    ):
        return all(_pure_neg_equiv(va, vb) for va, vb in zip(a.values, b.values))
    return False


def _extract_cmp_none_arg(node: ast.AST, op_cls: type) -> ast.AST | None:
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], op_cls) and len(node.comparators) == 1:
        if _is_const_none(node.comparators[0]):
            return node.left
        if _is_const_none(node.left):
            return node.comparators[0]
    return None


def analyze_diff_asts(
    old_tree: ast.AST,
    new_tree: ast.AST,
    ignore_annotations: bool = False,
    block_class_ann: bool = False,
    collect_runtime_swaps: bool = True,
):
    cmp_swaps = []
    falsy_swaps = []
    dict_diffs = []
    other_spans = []

    def walk_cmp(o: ast.AST, n: ast.AST, cls_stack: list[str], in_func: bool, bool_ctx: bool) -> bool:
        # 1. Pure boolean negation / operator inversion (symmetric, 0 tests)
        if _unwrap_not_cmp_invert(o, n) or _unwrap_not_cmp_invert(n, o):
            return True

        # 2. De Morgan: not (a or b) <-> not a and not b (always returns bool, short-circuits identically)
        if isinstance(o, ast.UnaryOp) and isinstance(o.op, ast.Not) and isinstance(o.operand, ast.BoolOp) and isinstance(n, ast.BoolOp):
            if type(o.operand.op) is not type(n.op) and len(o.operand.values) == len(n.values):
                if all(_pure_neg_equiv(ov, nv) for ov, nv in zip(o.operand.values, n.values)):
                    return True
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not) and isinstance(n.operand, ast.BoolOp) and isinstance(o, ast.BoolOp):
            if type(n.operand.op) is not type(o.op) and len(n.operand.values) == len(o.values):
                if all(_pure_neg_equiv(ov, nv) for ov, nv in zip(o.values, n.operand.values)):
                    return True

        if collect_runtime_swaps:
            # 3. x is None <-> not x (returns bool in both forms)
            ox_isnone = _extract_cmp_none_arg(o, ast.Is)
            if ox_isnone is not None and isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not) and _can_be_bool_none(n.operand):
                if ast_equal(ox_isnone, n.operand):
                    falsy_swaps.append((o.lineno, o.col_offset, "Is"))
                    return True
            nx_isnone = _extract_cmp_none_arg(n, ast.Is)
            if nx_isnone is not None and isinstance(o, ast.UnaryOp) and isinstance(o.op, ast.Not) and _can_be_bool_none(o.operand):
                if ast_equal(o.operand, nx_isnone):
                    falsy_swaps.append((o.operand.lineno, o.operand.col_offset, "BoolNone"))
                    return True

            # 4. x is not None <-> x (strictly when bool_ctx=True)
            if bool_ctx:
                ox_isnot = _extract_cmp_none_arg(o, ast.IsNot)
                if ox_isnot is not None and _can_be_bool_none(n) and ast_equal(ox_isnot, n):
                    falsy_swaps.append((o.lineno, o.col_offset, "IsNot"))
                    return True
                nx_isnot = _extract_cmp_none_arg(n, ast.IsNot)
                if nx_isnot is not None and _can_be_bool_none(o) and ast_equal(o, nx_isnot):
                    falsy_swaps.append((o.lineno, o.col_offset, "BoolNone"))
                    return True

        # 5. Local AnnAssign <-> Assign inside functions when annotations ignored
        if ignore_annotations and in_func:
            if isinstance(o, ast.AnnAssign) and isinstance(n, ast.Assign) and o.value is not None and len(n.targets) == 1:
                if ast_equal(o.target, n.targets[0]):
                    return walk_cmp(o.value, n.value, cls_stack, True, False)
            if isinstance(o, ast.Assign) and isinstance(n, ast.AnnAssign) and n.value is not None and len(o.targets) == 1:
                if ast_equal(o.targets[0], n.target):
                    return walk_cmp(o.value, n.value, cls_stack, True, False)

        if type(o) is not type(n):
            return False

        if isinstance(o, ast.ClassDef):
            obody = _filter_tc_stmts(o.body)
            nbody = _filter_tc_stmts(n.body)
            if (
                o.name != n.name
                or len(obody) != len(nbody)
                or not ast_equal(o.bases, n.bases)
                or not ast_equal(o.keywords, n.keywords)
                or not ast_equal(o.decorator_list, n.decorator_list)
            ):
                return False
            cls_stack.append(o.name)
            for ob, nb in zip(obody, nbody):
                if not walk_cmp(ob, nb, cls_stack, False, False):
                    cls_stack.pop()
                    return False
            cls_stack.pop()
            return True

        if collect_runtime_swaps and cls_stack and isinstance(o, (ast.Assign, ast.AnnAssign)) and isinstance(o.value, ast.Dict) and isinstance(n.value, ast.Dict):
            target_id = None
            if isinstance(o, ast.Assign) and len(o.targets) == len(n.targets) == 1:
                if isinstance(o.targets[0], ast.Name) and isinstance(n.targets[0], ast.Name) and o.targets[0].id == n.targets[0].id:
                    target_id = o.targets[0].id
            elif isinstance(o, ast.AnnAssign):
                if (
                    isinstance(o.target, ast.Name)
                    and isinstance(n.target, ast.Name)
                    and o.target.id == n.target.id
                    and (ignore_annotations and not block_class_ann or ast_equal(o.annotation, n.annotation))
                ):
                    target_id = o.target.id
            if target_id is not None:
                res = _diff_dict_nodes(o.value, n.value)
                if res is not None:
                    changed_keys, lambda_pairs = res
                    if changed_keys:
                        dict_diffs.append((".".join(cls_stack), target_id, o.lineno, o.end_lineno or o.lineno, changed_keys))
                    for ob, nb in lambda_pairs:
                        if not walk_cmp(ob, nb, cls_stack, True, False):
                            return False
                    return True

        if collect_runtime_swaps and isinstance(o, ast.Compare) and len(o.ops) == 1 and len(n.ops) == 1:
            op_pair = (type(o.ops[0]).__name__, type(n.ops[0]).__name__)
            if op_pair in REVERSIBLE_CMPS:
                if ast_equal(o.left, n.left) and len(o.comparators) == len(n.comparators) == 1 and ast_equal(o.comparators[0], n.comparators[0]):
                    if op_pair[0] in ("Eq", "NotEq"):
                        if not (_is_const_none_or_bool(o.left) or _is_const_none_or_bool(o.comparators[0])):
                            other_spans.append((o.lineno, o.end_lineno or o.lineno))
                            return True
                    cmp_swaps.append((o.lineno, o.col_offset, op_pair[0]))
                    return True
                other_spans.append((o.lineno, o.end_lineno or o.lineno))
                return True

        child_in_func = in_func or isinstance(o, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        for field in o._fields:
            if ignore_annotations:
                if isinstance(o, (ast.FunctionDef, ast.AsyncFunctionDef)) and field in ("returns", "type_params"):
                    continue
                if isinstance(o, ast.arg) and field == "annotation":
                    continue
                if isinstance(o, ast.AnnAssign) and field == "annotation":
                    if in_func or (o.value is not None and n.value is not None and not block_class_ann):
                        continue

            if isinstance(o, (ast.If, ast.While, ast.IfExp, ast.Assert)):
                child_bool = field == "test"
            elif isinstance(o, ast.UnaryOp) and isinstance(o.op, ast.Not):
                child_bool = field == "operand"
            elif isinstance(o, ast.comprehension):
                child_bool = field == "ifs"
            elif isinstance(o, ast.BoolOp):
                child_bool = bool_ctx
            else:
                child_bool = False

            ov = getattr(o, field)
            nv = getattr(n, field)
            if isinstance(ov, list) and isinstance(nv, list):
                if field in ("body", "orelse", "finalbody"):
                    ov = _filter_tc_stmts(ov)
                    nv = _filter_tc_stmts(nv)
                if len(ov) != len(nv):
                    return False
                for oi, ni in zip(ov, nv):
                    if isinstance(oi, ast.AST) and isinstance(ni, ast.AST):
                        if not walk_cmp(oi, ni, cls_stack, child_in_func, child_bool):
                            return False
                    elif oi != ni:
                        if hasattr(o, "lineno"):
                            other_spans.append((o.lineno, getattr(o, "end_lineno", None) or o.lineno))
                            return True
                        return False
            elif isinstance(ov, ast.AST) and isinstance(nv, ast.AST):
                if not walk_cmp(ov, nv, cls_stack, child_in_func, child_bool):
                    return False
            elif ov != nv:
                if hasattr(o, "lineno"):
                    other_spans.append((o.lineno, getattr(o, "end_lineno", None) or o.lineno))
                    return True
                return False
        return True

    ok = walk_cmp(old_tree, new_tree, [], False, False)
    return cmp_swaps, falsy_swaps, dict_diffs, other_spans, not ok


def _only_added_new_test_methods(old_tree: ast.AST, new_tree: ast.AST, rel: str, id_to_num: dict[str, int]) -> bool:
    if not (isinstance(old_tree, ast.Module) and isinstance(new_tree, ast.Module)):
        return False
    ob = _filter_tc_stmts(old_tree.body)
    nb = _filter_tc_stmts(new_tree.body)
    if len(ob) != len(nb):
        return False
    added_any = False
    for os_n, ns_n in zip(ob, nb):
        if isinstance(os_n, ast.ClassDef) and isinstance(ns_n, ast.ClassDef) and os_n.name == ns_n.name:
            if not (ast_equal(os_n.bases, ns_n.bases) and ast_equal(os_n.keywords, ns_n.keywords) and ast_equal(os_n.decorator_list, ns_n.decorator_list)):
                return False
            oc = _filter_tc_stmts(os_n.body)
            nc = _filter_tc_stmts(ns_n.body)
            oi = 0
            for item in nc:
                if oi < len(oc) and ast_equal(oc[oi], item):
                    oi += 1
                elif isinstance(item, ast.FunctionDef) and item.name.startswith("test_") and f"{rel}::{ns_n.name}::{item.name}" not in id_to_num:
                    added_any = True
                else:
                    return False
            if oi != len(oc):
                return False
        elif not ast_equal(os_n, ns_n):
            return False
    return added_any


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


def _load_root(path: Path) -> dict:
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

    maps = _load_root(MAPS_PATH)
    is_split = bool(maps.get("split"))
    maps_dir = MAPS_PATH.parent

    idx_to_num_list = [id_to_num.get(nid, -1) for nid in maps["nodeids"]]

    def remap_idxs(idxs) -> set[int]:
        if not idxs:
            return set()
        out = set()
        for i in idxs:
            num = idx_to_num_list[i]
            if num >= 0:
                out.add(num)
        return out

    # Lazy remapping cache for small top-level dialect/fixture dicts
    def make_lazy_dict(raw: dict) -> dict[str, set[int]]:
        cache: dict[str, set[int]] = {}
        class LazyMap(dict):
            def get(self, key, default=None):
                if key in cache:
                    return cache[key]
                v = raw.get(key)
                if v is None:
                    return default
                res = remap_idxs(v)
                cache[key] = res
                return res
            def __contains__(self, key):
                return key in raw
            def __getitem__(self, key):
                res = self.get(key)
                if res is None:
                    raise KeyError(key)
                return res
        return LazyMap()

    fixture_to_tests = make_lazy_dict(maps["fixture_to_idxs"])
    dialect_to_tests = make_lazy_dict(maps["dialect_to_idxs"])
    tok_to_tests = make_lazy_dict(maps.get("tok_to_idxs", {}))
    parse_to_tests = make_lazy_dict(maps.get("parse_to_idxs", {}))
    gen_init_to_tests = make_lazy_dict(maps.get("gen_init_to_idxs", {}))
    gen_sql_to_tests = make_lazy_dict(maps.get("gen_sql_to_idxs", {}))

    dialect_subclasses = maps["dialect_subclasses"]
    parser_subclasses = maps["parser_subclasses"]
    generator_subclasses = maps["generator_subclasses"]
    tokenizer_subclasses = maps.get("tokenizer_subclasses", {})
    mod_attr_readers = maps.get("mod_attr_readers", {})
    mod_name_readers = maps.get("mod_name_readers", {})
    file_deps = maps.get("file_deps", {})
    cmp_diff_to_tests = make_lazy_dict(maps.get("cmp_diff_to_idxs", {}))
    scalar_attr_to_tests = make_lazy_dict(maps.get("scalar_attr_to_idxs", {}))
    cls_sub_qualnames = maps.get("cls_sub_qualnames", {})
    tracked_dicts = {tuple(x) for x in maps.get("tracked_dicts", ())}
    test_dict_readers = maps.get("test_dict_readers", {})
    scalar_attrs = set(maps.get("scalar_attrs", ()))

    if is_split:
        file_ids = maps["file_ids"]
        ast_hashes = maps["ast_hashes"]
        test_methods_map = maps["test_methods"]
        _readers_cache: dict | None = None
        _dict_keys_cache: dict | None = None
        _falsy_diffs_cache: dict | None = None

        def get_readers() -> tuple[dict, dict]:
            nonlocal _readers_cache
            if _readers_cache is None:
                with open(maps_dir / "readers.pkl", "rb") as rf:
                    _readers_cache = pickle.load(rf)
            return _readers_cache["attr_readers"], _readers_cache["name_readers"]

        def get_dict_keys() -> dict:
            nonlocal _dict_keys_cache
            if _dict_keys_cache is None:
                dk_path = maps_dir / "dict_keys.pkl"
                if dk_path.exists():
                    with open(dk_path, "rb") as dk:
                        _dict_keys_cache = pickle.load(dk)
                else:
                    _dict_keys_cache = {}
            return _dict_keys_cache

        def get_falsy_diffs() -> dict:
            nonlocal _falsy_diffs_cache
            if _falsy_diffs_cache is None:
                fd_path = maps_dir / "falsy_diffs.pkl"
                if fd_path.exists():
                    with open(fd_path, "rb") as fd:
                        _falsy_diffs_cache = pickle.load(fd)
                else:
                    _falsy_diffs_cache = {}
            return _falsy_diffs_cache

        _meta_cache: dict[str, dict] = {}

        def get_meta(rel: str) -> dict | None:
            if rel in _meta_cache:
                return _meta_cache[rel]
            fid = file_ids.get(rel)
            if fid is None:
                _meta_cache[rel] = None
                return None
            with open(maps_dir / "meta" / f"{fid}.pkl", "rb") as mf:
                m = pickle.load(mf)
            _meta_cache[rel] = m
            return m

        def has_file(rel: str) -> bool:
            return rel in file_ids

        def get_ast_hash(rel: str) -> str | None:
            return ast_hashes.get(rel)

        def get_test_methods(rel: str) -> list:
            return test_methods_map.get(rel, ())

        def get_line_tests(rel: str, ln: int) -> set[int]:
            m = get_meta(rel)
            if not m:
                return set()
            return remap_idxs(m["line_to_idxs"].get(ln))

        def get_file_cov(rel: str) -> set[int]:
            m = get_meta(rel)
            if not m:
                return set()
            return remap_idxs(m["file_cov_idxs"])

        def get_line_info(rel: str, ln: int) -> dict:
            m = get_meta(rel)
            if not m:
                return {}
            li = m["line_info"]
            return li.get(ln) or li.get(str(ln), {})

        def get_stmt_span(rel: str, ln: int) -> list[int] | None:
            m = get_meta(rel)
            if not m:
                return None
            ss = m["stmt_spans"]
            return ss.get(ln) or ss.get(str(ln))

        def get_str_lines(rel: str):
            m = get_meta(rel)
            return m.get("str_lines", ()) if m else ()
    else:
        file_data = maps["file_data"]
        legacy_attr_readers = maps["attr_readers"]
        legacy_name_readers = maps["name_readers"]
        legacy_lines = maps["line_to_idxs"]
        legacy_file_cov: dict[str, list[int]] = defaultdict(list)
        for key, idxs in legacy_lines.items():
            rel_k, _ = key.rsplit(":", 1)
            legacy_file_cov[rel_k].extend(idxs)

        def get_readers() -> tuple[dict, dict]:
            return legacy_attr_readers, legacy_name_readers

        def get_dict_keys() -> dict:
            return {}

        def get_falsy_diffs() -> dict:
            return {}

        def has_file(rel: str) -> bool:
            return rel in file_data

        def get_ast_hash(rel: str) -> str | None:
            fd = file_data.get(rel)
            return fd.get("ast_hash") if fd else None

        def get_test_methods(rel: str) -> list:
            fd = file_data.get(rel)
            return fd.get("test_methods", ()) if fd else ()

        def get_line_tests(rel: str, ln: int) -> set[int]:
            return remap_idxs(legacy_lines.get(f"{rel}:{ln}"))

        def get_file_cov(rel: str) -> set[int]:
            return remap_idxs(legacy_file_cov.get(rel))

        def get_line_info(rel: str, ln: int) -> dict:
            fd = file_data.get(rel)
            if not fd:
                return {}
            li = fd["line_info"]
            return li.get(ln) or li.get(str(ln), {})

        def get_stmt_span(rel: str, ln: int) -> list[int] | None:
            fd = file_data.get(rel)
            if not fd:
                return None
            ss = fd["stmt_spans"]
            return ss.get(ln) or ss.get(str(ln))

        def get_str_lines(rel: str):
            fd = file_data.get(rel)
            return fd.get("str_lines", ()) if fd else ()

    # Build transitive importer maps from file_deps (~100 entries, < 0.5ms)
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

    orig_src_cache: dict[str, str] = {}
    orig_ast_cache: dict[str, ast.AST] = {}

    def get_orig(rel: str) -> str:
        if rel not in orig_src_cache:
            p = REF / rel
            orig_src_cache[rel] = p.read_text("utf-8") if p.exists() else ""
        return orig_src_cache[rel]

    def get_orig_ast(rel: str, orig_src: str) -> ast.AST:
        t = orig_ast_cache.get(rel)
        if t is None:
            t = ast.parse(orig_src)
            orig_ast_cache[rel] = t
        return t

    def tests_for_line_or_span(rel: str, ln: int) -> set[int]:
        out = get_line_tests(rel, ln)
        span = get_stmt_span(rel, ln)
        if span:
            for sln in range(span[0], span[1] + 1):
                if sln != ln:
                    out.update(get_line_tests(rel, sln))
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
            r_info = get_line_info(r_rel, r_ln)
            if r_info.get("func") == "__init__":
                source = gen_init_to_tests or dialect_to_tests
            else:
                source = gen_sql_to_tests or dialect_to_tests
        else:
            source = dialect_to_tests
        res: set[int] = set()
        for dn in d_names:
            t = source.get(dn)
            if t:
                res.update(t)
        return res

    def union_dialect_tests(d_names: list[str]) -> set[int]:
        res: set[int] = set()
        for dn in d_names:
            t = dialect_to_tests.get(dn)
            if t:
                res.update(t)
        return res

    def tests_for_test_line(r_rel: str, r_ln: int) -> set[int]:
        for cls_n, meth_n, s_ln, e_ln in get_test_methods(r_rel):
            if s_ln <= r_ln <= e_ln:
                tnum = id_to_num.get(f"{r_rel}::{cls_n}::{meth_n}")
                if tnum is not None:
                    return {tnum}
        if has_file(r_rel):
            r_info = get_line_info(r_rel, r_ln)
            if r_info.get("is_lambda") or r_info.get("func"):
                fn = r_info.get("func")
                if fn and (fn in ONCE_FUNCS or fn in ("setUpClass", "tearDownClass", "setUpModule", "tearDownModule")):
                    return set(tests_by_file.get(r_rel, ())) or file_importers.get(r_rel, set())
                return tests_for_line_or_span(r_rel, r_ln)
        return set(tests_by_file.get(r_rel, ())) or file_importers.get(r_rel, set())

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

                fx_tests = fixture_to_tests.get(rel)
                if fx_tests:
                    selected.update(fx_tests)
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
                default_test_fallback = file_importers.get(rel, set() if rel in file_deps else all_test_nums)
                if not rel.endswith(".py") or fdiff["new_file"] or fdiff["deleted_file"]:
                    selected.update(default_test_fallback)
                    continue

                if has_file(rel) and is_comment_or_blank_only(fdiff["hunks"], get_str_lines(rel)):
                    continue

                t_other_spans: list[tuple[int, int]] = []
                t_mismatch = False
                t_use_ast = False

                orig_src = get_orig(rel)
                patched_src = apply_hunks(orig_src, fdiff["hunks"]) if orig_src else None
                if patched_src is not None and orig_src:
                    try:
                        new_ast = ast.parse(patched_src)
                    except SyntaxError:
                        selected.update(default_test_fallback)
                        continue
                    old_ast = get_orig_ast(rel, orig_src)
                    if ast_equal(old_ast, new_ast):
                        continue
                    _, _, _, t_other_spans, t_mismatch = analyze_diff_asts(
                        old_ast,
                        new_ast,
                        ignore_annotations=has_future_annotations(old_ast),
                        collect_runtime_swaps=False,
                    )
                    if not t_mismatch and not t_other_spans:
                        continue
                    if _only_added_new_test_methods(old_ast, new_ast, rel, id_to_num):
                        continue
                    t_use_ast = True

                test_methods = get_test_methods(rel)
                if rel in tests_by_file and test_methods:
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
                        if t_use_ast and not t_mismatch:
                            touched = [ln for ln in touched if any(s <= ln <= e for s, e in t_other_spans)]
                            inserts = [ln for ln in inserts if any((s - 1) <= ln <= (e + 1) for s, e in t_other_spans)]
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

                if has_file(rel) and rel not in ("tests/gen_fixtures.py", "tests/leakcheck.py"):
                    importers_of_rel = mod_importers.get(rel, {rel})
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
                        if t_use_ast and not t_mismatch:
                            touched_old_lines = [ln for ln in touched_old_lines if any(s <= ln <= e for s, e in t_other_spans)]
                            insertion_anchors = [ln for ln in insertion_anchors if any((s - 1) <= ln <= (e + 1) for s, e in t_other_spans)]
                        candidate_lines = touched_old_lines if touched_old_lines else insertion_anchors
                        for ln in candidate_lines:
                            if ln < 1:
                                continue
                            info = get_line_info(rel, ln)
                            if info.get("is_lambda"):
                                lr = info["lambda_range"]
                                for l2 in range(lr[0], lr[1] + 1):
                                    selected.update(tests_for_line_or_span(rel, l2))
                            elif info.get("func"):
                                fn = info["func"]
                                if fn in ONCE_FUNCS or fn in ("setUpClass", "tearDownClass", "setUpModule", "tearDownModule"):
                                    selected.update(default_test_fallback)
                                    continue
                                cov_here = tests_for_line_or_span(rel, ln)
                                if cov_here:
                                    selected.update(cov_here)
                                else:
                                    fr = info["func_range"]
                                    sig_end = info.get("sig_end", fr[0] + 1)
                                    if ln <= sig_end:
                                        for l2 in range(fr[0], fr[1] + 1):
                                            selected.update(get_line_tests(rel, l2))
                                    elif not touched_old_lines:
                                        for delta in (-1, 1, -2, 2):
                                            selected.update(tests_for_line_or_span(rel, ln + delta))
                                    if fn:
                                        _, name_readers = get_readers()
                                        for mrel in mod_name_readers.get(fn, ()):
                                            if mrel in importers_of_rel:
                                                selected.update(file_importers.get(mrel, ()))
                                        for loc in name_readers.get(fn, ()):
                                            r_rel, r_ln_str = loc.rsplit(":", 1)
                                            if r_rel in importers_of_rel:
                                                r_ln = int(r_ln_str)
                                                r_fn = get_line_info(r_rel, r_ln).get("func")
                                                if r_fn in ("setUpClass", "tearDownClass"):
                                                    selected.update(tests_for_test_line(r_rel, r_ln))
                            elif "module_var" in info or "class_attr" in info:
                                attr_readers, name_readers = get_readers()
                                if "module_var" in info:
                                    sym = info["module_var"]
                                    cls_s = None
                                else:
                                    cls_s, sym = info["class_attr"]
                                matched_any_reader = False
                                if sym:
                                    for loc in name_readers.get(sym, ()):
                                        r_rel, r_ln_str = loc.rsplit(":", 1)
                                        if r_rel in importers_of_rel:
                                            matched_any_reader = True
                                            selected.update(tests_for_test_line(r_rel, int(r_ln_str)))
                                    for loc in attr_readers.get(sym, ()):
                                        r_rel, r_ln_str = loc.rsplit(":", 1)
                                        if r_rel in importers_of_rel:
                                            matched_any_reader = True
                                            selected.update(tests_for_test_line(r_rel, int(r_ln_str)))
                                    for mrel in mod_name_readers.get(sym, ()):
                                        if mrel in importers_of_rel:
                                            matched_any_reader = True
                                            selected.update(file_importers.get(mrel, ()))
                                    for mrel in mod_attr_readers.get(sym, ()):
                                        if mrel in importers_of_rel:
                                            matched_any_reader = True
                                            selected.update(file_importers.get(mrel, ()))
                                if not matched_any_reader and sym is None:
                                    selected.update(default_test_fallback)
                            else:
                                selected.update(default_test_fallback)
                    continue

                selected.update(default_test_fallback)
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

            if has_file(rel) and is_comment_or_blank_only(fdiff["hunks"], get_str_lines(rel)):
                continue

            orig_src = get_orig(rel)
            patched_src = apply_hunks(orig_src, fdiff["hunks"]) if orig_src else None
            cmp_swaps = []
            falsy_swaps = []
            dict_diffs = []
            other_spans: list[tuple[int, int]] = []
            mismatch = False
            use_ast_diff = False

            if patched_src is not None and orig_src:
                try:
                    new_ast = ast.parse(patched_src)
                except SyntaxError:
                    selected.update(file_importers.get(rel, all_test_nums))
                    continue
                old_ast = get_orig_ast(rel, orig_src)
                ignore_ann = (rel not in _RUNTIME_ANN_FILES) and has_future_annotations(old_ast)
                block_class_ann = rel in _CLASS_ANN_FILES
                cmp_swaps, falsy_swaps, dict_diffs, other_spans, mismatch = analyze_diff_asts(
                    old_ast,
                    new_ast,
                    ignore_annotations=ignore_ann,
                    block_class_ann=block_class_ann,
                )
                if not mismatch and not cmp_swaps and not falsy_swaps and not dict_diffs and not other_spans:
                    continue
                use_ast_diff = True

            if rel.startswith("sqlglot/typing/"):
                selected.update(tests_by_file.get("tests/test_typing.py", ()))

            if not has_file(rel):
                selected.update(all_test_nums)
                continue

            if use_ast_diff and not mismatch:
                for ln, col, op_name in cmp_swaps:
                    linfo = get_line_info(rel, ln)
                    if (linfo.get("func") and linfo["func"] not in ONCE_FUNCS) or linfo.get("is_lambda"):
                        selected.update(cmp_diff_to_tests.get(f"{rel}:{ln}:{col}:{op_name}", ()))
                    else:
                        other_spans.append((ln, ln))

                if falsy_swaps:
                    fd_map = get_falsy_diffs()
                    for ln, col, kind in falsy_swaps:
                        linfo = get_line_info(rel, ln)
                        if (linfo.get("func") and linfo["func"] not in ONCE_FUNCS) or linfo.get("is_lambda"):
                            idxs = fd_map.get(f"{rel}:{ln}:{col}:{kind}")
                            if idxs:
                                selected.update(remap_idxs(idxs))
                        else:
                            other_spans.append((ln, ln))

                if dict_diffs:
                    dk_map = None
                    for cls_name, attr_name, d_ln, d_end_ln, changed_keys in dict_diffs:
                        if (cls_name, attr_name) in tracked_dicts:
                            if dk_map is None:
                                dk_map = get_dict_keys()
                            sub_cls_list = cls_sub_qualnames.get(cls_name, (cls_name,))
                            for sub_c in sub_cls_list:
                                full_idxs = dk_map.get(f"{sub_c}|{attr_name}|__FULL__")
                                if full_idxs:
                                    selected.update(remap_idxs(full_idxs))
                                for kt in changed_keys:
                                    k_idxs = dk_map.get(f"{sub_c}|{attr_name}|{kt}")
                                    if k_idxs:
                                        selected.update(remap_idxs(k_idxs))
                            for loc in test_dict_readers.get(attr_name, ()):
                                r_rel, r_ln_str = loc.rsplit(":", 1)
                                selected.update(tests_for_test_line(r_rel, int(r_ln_str)))
                        else:
                            other_spans.append((d_ln, d_end_ln))

                if not other_spans:
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

                if use_ast_diff and not mismatch:
                    touched_old_lines = [
                        ln for ln in touched_old_lines if any(s <= ln <= e for s, e in other_spans)
                    ]
                    insertion_anchors = [
                        ln for ln in insertion_anchors if any((s - 1) <= ln <= (e + 1) for s, e in other_spans)
                    ]

                candidate_lines = touched_old_lines if touched_old_lines else insertion_anchors
                for ln in candidate_lines:
                    if ln < 1:
                        continue
                    info = get_line_info(rel, ln)
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
                            sig_end = info.get("sig_end", fr[0] + 1)
                            if ln <= sig_end:
                                for l2 in range(fr[0], fr[1] + 1):
                                    selected.update(get_line_tests(rel, l2))
                            elif not touched_old_lines:
                                for delta in (-1, 1, -2, 2):
                                    selected.update(tests_for_line_or_span(rel, ln + delta))
                    elif "class_attr" in info:
                        attr_readers, name_readers = get_readers()
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

                        def tests_for_reader_loc(r_rel: str, r_ln: int) -> set[int]:
                            direct = get_line_tests(r_rel, r_ln)
                            return direct if direct else tests_for_line_or_span(r_rel, r_ln)

                        if d_names is not None:
                            if reader_locs:
                                sqlglot_matched = set()
                                has_sqlglot_reader = False
                                test_matched = set()
                                for loc in reader_locs:
                                    r_rel, r_ln_str = loc.rsplit(":", 1)
                                    r_ln = int(r_ln_str)
                                    if r_rel.startswith("tests/"):
                                        test_matched.update(tests_for_test_line(r_rel, r_ln))
                                    else:
                                        has_sqlglot_reader = True
                                        t_here = tests_for_reader_loc(r_rel, r_ln)
                                        if t_here:
                                            sqlglot_matched.update(t_here & component_tests_for_reader(d_names, r_rel, r_ln))
                                if has_sqlglot_reader and attr_name in scalar_attrs:
                                    sc_tests = set()
                                    sc_keys = set(d_names)
                                    if cls_name:
                                        sc_keys.update(cls_sub_qualnames.get(cls_name, (cls_name,)))
                                    for k in sc_keys:
                                        t = scalar_attr_to_tests.get(f"{k}|{attr_name}")
                                        if t:
                                            sc_tests.update(t)
                                    selected.update(sqlglot_matched & sc_tests)
                                else:
                                    selected.update(sqlglot_matched)
                                selected.update(test_matched)
                            else:
                                selected.update(union_dialect_tests(d_names))
                            if attr_name:
                                for mrel in mod_attr_readers.get(attr_name, ()):
                                    if mrel in importers_of_rel:
                                        selected.update(file_importers.get(mrel, ()))
                        elif reader_locs:
                            for loc in reader_locs:
                                r_rel, r_ln_str = loc.rsplit(":", 1)
                                r_ln = int(r_ln_str)
                                if r_rel.startswith("tests/"):
                                    selected.update(tests_for_test_line(r_rel, r_ln))
                                else:
                                    selected.update(tests_for_reader_loc(r_rel, r_ln))
                            if attr_name:
                                mod_attr_set = set(mod_attr_readers.get(attr_name, ()))
                                if is_expr_class_attr and cls_name:
                                    mod_cls_set = set(mod_attr_readers.get(cls_name, ())) | set(mod_name_readers.get(cls_name, ()))
                                    mod_matches = mod_attr_set & mod_cls_set & importers_of_rel
                                else:
                                    mod_matches = mod_attr_set & importers_of_rel
                                for mrel in mod_matches:
                                    selected.update(file_importers.get(mrel, ()))
                        elif not is_expr_class_attr:
                            selected.update(get_file_cov(rel))
                    elif "module_var" in info:
                        attr_readers, name_readers = get_readers()
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
                                r_ln = int(r_ln_str)
                                if r_rel.startswith("tests/"):
                                    selected.update(tests_for_test_line(r_rel, r_ln))
                                else:
                                    selected.update(tests_for_line_or_span(r_rel, r_ln))
                        elif not (var_name and (mod_name_readers.get(var_name) or mod_attr_readers.get(var_name))):
                            selected.update(get_file_cov(rel))
                    else:
                        selected.update(get_file_cov(rel))

        return selected

    for patch in sorted((DATA / "commits").glob("*.patch")):
        sel = select_for_patch(patch.read_text("utf-8"))
        workers = pack_workers(sel, dur_ms)
        sys.stdout.write(json.dumps({"commit": patch.stem, "workers": workers}) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
