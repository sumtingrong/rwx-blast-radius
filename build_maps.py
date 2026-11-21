"""Build-time coverage, dialect-usage, fixture, and static AST map generator for sqlglot.

Designed for Python 3.13 (uses sys.monitoring + multiprocessing) inside the Linux Docker build,
and also works on Python 3.11+ via sys.settrace fallback.
"""
from __future__ import annotations

import ast
import builtins
import concurrent.futures
import gzip
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import multiprocessing
import os
import pickle
import sys
import types
import unittest
from collections import defaultdict
from concurrent.futures import Future
from pathlib import Path

DERIVED_SOURCE_ATTRS = frozenset({
    "TRANSFORMS",
    "COERCES_TO",
    "TIME_MAPPING",
    "FORMAT_MAPPING",
    "CREATABLE_KIND_MAPPING",
    "UNESCAPED_SEQUENCES",
    "DATE_PART_MAPPING",
    "VECTOR_TYPE_ALIASES",
    "QUOTES",
    "IDENTIFIERS",
    "COMMENTS",
    "STRING_ESCAPES",
    "BYTE_STRING_ESCAPES",
    "IDENTIFIER_ESCAPES",
    "SUPPORTED_JSON_PATH_PARTS",
    "HINT_START",
    "BIT_STRINGS",
    "BYTE_STRINGS",
    "HEX_STRINGS",
    "NATIONAL_RAW_STRINGS",
    "RAW_STRINGS",
    "HEREDOC_STRINGS",
    "UNICODE_STRINGS",
})

_SAFE_NONE_EQ_TYPES = (str, int, float, bool, list, dict, tuple, set, frozenset, bytes)


def _key_tag(k) -> str | None:
    if isinstance(k, (str, int, float, bool)) or k is None:
        return f"s:{k!r}"
    if isinstance(k, type):
        return f"t:{k.__qualname__}"
    tp = type(k)
    if hasattr(k, "name") and hasattr(tp, "__members__"):
        return f"e:{tp.__name__}.{k.name}"
    return None


class TrackedDict(dict):
    _rec_fn = None

    def __init__(self, data, tag: tuple[str, str]):
        super().__init__(data)
        self._tag = tag

    def get(self, key, default=None):
        rec = TrackedDict._rec_fn
        if rec is not None:
            kt = _key_tag(key)
            rec(self._tag, kt if kt is not None else "__FULL__")
        return super().get(key, default)

    def __getitem__(self, key):
        rec = TrackedDict._rec_fn
        if rec is not None:
            kt = _key_tag(key)
            rec(self._tag, kt if kt is not None else "__FULL__")
        return super().__getitem__(key)

    def items(self):
        rec = TrackedDict._rec_fn
        if rec is not None:
            rec(self._tag, "__FULL__")
        return super().items()

    def values(self):
        rec = TrackedDict._rec_fn
        if rec is not None:
            rec(self._tag, "__FULL__")
        return super().values()

    def __contains__(self, key):
        rec = TrackedDict._rec_fn
        if rec is not None:
            kt = _key_tag(key)
            rec(self._tag, kt if kt is not None else "__FULL__")
        return super().__contains__(key)

    def keys(self):
        rec = TrackedDict._rec_fn
        if rec is not None:
            rec(self._tag, "__FULL__")
        return super().keys()

    def __iter__(self):
        rec = TrackedDict._rec_fn
        if rec is not None:
            rec(self._tag, "__FULL__")
        return super().__iter__()

    def copy(self):
        rec = TrackedDict._rec_fn
        if rec is not None:
            rec(self._tag, "__FULL__")
        return super().copy()

    def __or__(self, other):
        rec = TrackedDict._rec_fn
        if rec is not None:
            rec(self._tag, "__FULL__")
        return super().__or__(other)

    def __ror__(self, other):
        rec = TrackedDict._rec_fn
        if rec is not None:
            rec(self._tag, "__FULL__")
        return super().__ror__(other)

    def __reduce_ex__(self, protocol):
        rec = TrackedDict._rec_fn
        if rec is not None:
            rec(self._tag, "__FULL__")
        return (dict, (dict(self),))


def iter_stmt_spans(node: ast.AST):
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body_start = child.body[0].lineno if child.body else child.end_lineno
            yield (child.lineno, max(child.lineno, body_start - 1), child)
            yield from iter_stmt_spans(child)
        elif isinstance(child, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)):
            if isinstance(child, (ast.If, ast.While)):
                yield (child.lineno, child.test.end_lineno or child.lineno, child)
            elif isinstance(child, (ast.For, ast.AsyncFor)):
                yield (child.lineno, child.iter.end_lineno or child.lineno, child)
            else:
                yield (child.lineno, child.lineno, child)
            yield from iter_stmt_spans(child)
        elif isinstance(child, ast.ExceptHandler):
            end_ln = child.type.end_lineno if child.type else child.lineno
            yield (child.lineno, end_ln or child.lineno, child)
            yield from iter_stmt_spans(child)
        elif isinstance(child, ast.stmt):
            yield (child.lineno, child.end_lineno or child.lineno, child)
            yield from iter_stmt_spans(child)
        elif isinstance(child, ast.Lambda):
            yield (child.lineno, child.end_lineno or child.lineno, child)
        else:
            yield from iter_stmt_spans(child)


def module_name_from_rel(rel: str) -> str:
    if rel.endswith("/__init__.py"):
        return rel[:-12].replace("/", ".")
    return rel[:-3].replace("/", ".")


def resolve_import(current_mod: str, is_pkg: bool, mod: str | None, level: int) -> list[str]:
    if level == 0:
        return [mod] if mod else []
    parts = current_mod.split(".") if is_pkg else current_mod.split(".")[:-1]
    if level > 1:
        parts = parts[: -(level - 1)]
    base = ".".join(parts)
    if mod:
        return [f"{base}.{mod}" if base else mod]
    return [base] if base else []


def ast_digest(tree: ast.AST) -> str:
    return hashlib.blake2b(ast.dump(tree).encode("utf-8"), digest_size=16).hexdigest()


def analyze_python_file(rel: str, src: str) -> dict:
    tree = ast.parse(src)
    mod_name = module_name_from_rel(rel)
    is_pkg = rel.endswith("/__init__.py")
    src_lines = src.splitlines()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(("sqlglot", "tests")):
                    imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            for base_mod in resolve_import(mod_name, is_pkg, node.module, node.level):
                if base_mod.startswith(("sqlglot", "tests")):
                    imports.add(base_mod)
                    for alias in node.names:
                        if alias.name != "*":
                            imports.add(f"{base_mod}.{alias.name}")
                            if base_mod == "sqlglot.dialects":
                                imports.add(f"sqlglot.dialects.{alias.name.lower()}")

    attr_reads = defaultdict(set)
    name_reads = defaultdict(set)
    mod_attr_reads: set[str] = set()
    mod_name_reads: set[str] = set()

    def walk_reads(node: ast.AST, in_func: bool) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for ch in ast.iter_child_nodes(node):
                walk_reads(ch, True)
            return
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            if in_func:
                attr_reads[node.attr].add(node.lineno)
            else:
                mod_attr_reads.add(node.attr)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if in_func:
                name_reads[node.id].add(node.lineno)
            else:
                mod_name_reads.add(node.id)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            if in_func:
                attr_reads[node.args[1].value].add(node.lineno)
            else:
                mod_attr_reads.add(node.args[1].value)
        for ch in ast.iter_child_nodes(node):
            walk_reads(ch, in_func)

    walk_reads(tree, False)

    line_info: dict[int, dict] = {}

    def mark_lines(start: int, end: int | None, **kwargs) -> None:
        for ln in range(start, (end or start) + 1):
            line_info.setdefault(ln, {}).update(kwargs)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start_ln = node.decorator_list[0].lineno if node.decorator_list else node.lineno
            body_start = node.body[0].lineno if node.body else node.lineno
            sig_end = max(node.lineno, body_start - 1)
            mark_lines(
                start_ln,
                node.end_lineno,
                func=node.name,
                func_range=[start_ln, node.end_lineno],
                sig_end=sig_end,
            )

    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            mark_lines(node.lineno, node.end_lineno, is_lambda=True, lambda_range=[node.lineno, node.end_lineno])

    def mark_class_body(cls_node: ast.ClassDef, qual_name: str) -> None:
        for item in cls_node.body:
            if isinstance(item, ast.ClassDef):
                mark_class_body(item, f"{qual_name}.{item.name}")
            elif isinstance(item, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                target_name = None
                if isinstance(item, ast.Assign) and item.targets and isinstance(item.targets[0], ast.Name):
                    target_name = item.targets[0].id
                elif isinstance(item, (ast.AnnAssign, ast.AugAssign)) and isinstance(item.target, ast.Name):
                    target_name = item.target.id
                for ln in range(item.lineno, (item.end_lineno or item.lineno) + 1):
                    info = line_info.get(ln, {})
                    if not info.get("is_lambda") and not info.get("func"):
                        line_info.setdefault(ln, {})["class_attr"] = [qual_name, target_name]

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            mark_class_body(node, node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            target_name = None
            if isinstance(node, ast.Assign) and node.targets and isinstance(node.targets[0], ast.Name):
                target_name = node.targets[0].id
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name):
                target_name = node.target.id
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                info = line_info.get(ln, {})
                if not info.get("is_lambda") and not info.get("func"):
                    line_info.setdefault(ln, {})["module_var"] = target_name
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                line_info.setdefault(ln, {})["is_import"] = True

    stmt_spans: dict[int, list[int]] = {}
    for s, e, n in iter_stmt_spans(tree):
        if isinstance(n, ast.Lambda):
            for ln in range(s, e + 1):
                stmt_spans[ln] = [s, e]
        elif not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for ln in range(s, e + 1):
                prev = stmt_spans.get(ln)
                if prev is None or (e - s) <= (prev[1] - prev[0]):
                    stmt_spans[ln] = [s, e]

    # Map non-executing branch keyword lines (else:, try:, finally:) to their first body line
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try)):
            for block in (
                getattr(node, "orelse", ()),
                getattr(node, "finalbody", ()),
            ):
                if block:
                    first_stmt = block[0]
                    # Search upward from first_stmt.lineno - 1 for else: / finally:
                    for kw_ln in range(first_stmt.lineno - 1, node.lineno, -1):
                        stripped = src_lines[kw_ln - 1].strip()
                        if stripped in ("else:", "finally:"):
                            stmt_spans.setdefault(kw_ln, [kw_ln, first_stmt.lineno])
                            break
            if isinstance(node, ast.Try) and node.body:
                stmt_spans.setdefault(node.lineno, [node.lineno, node.body[0].lineno])

    test_methods = []
    if rel.startswith("tests/"):
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name != "Validator":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                        s_ln = item.decorator_list[0].lineno if item.decorator_list else item.lineno
                        e_ln = item.end_lineno or item.lineno
                        test_methods.append([node.name, item.name, s_ln, e_ln])

    str_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr) or (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            end_ln = node.end_lineno or node.lineno
            if end_ln > node.lineno:
                str_lines.update(range(node.lineno, end_ln + 1))

    return {
        "module": mod_name,
        "ast_hash": ast_digest(tree),
        "imports": sorted(imports),
        "attr_reads": {k: sorted(v) for k, v in attr_reads.items()},
        "name_reads": {k: sorted(v) for k, v in name_reads.items()},
        "mod_attr_reads": sorted(mod_attr_reads),
        "mod_name_reads": sorted(mod_name_reads),
        "line_info": line_info,
        "stmt_spans": stmt_spans,
        "test_methods": test_methods,
        "str_lines": str_lines,
    }


class _SyncPool:
    def __init__(self, *a, initializer=None, initargs=(), **kw):
        if initializer is not None:
            initializer(*initargs)

    def submit(self, fn, *args, **kwargs):
        fut = Future()
        try:
            p_args = pickle.loads(pickle.dumps(args))
            p_kwargs = pickle.loads(pickle.dumps(kwargs))
            res = fn(*p_args, **p_kwargs)
            res = pickle.loads(pickle.dumps(res))
            fut.set_result(res)
        except Exception as e:
            fut.set_exception(e)
        return fut

    def shutdown(self, wait=True, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _discover_unittest_items(repo_root: Path) -> list[tuple[str, str, str]]:
    """Discover all (rel_file, cls_name, method_name) in sqlglot/tests."""
    items = []
    tests_dir = repo_root / "tests"
    for p in sorted(tests_dir.rglob("test_*.py")):
        rel = p.relative_to(repo_root).as_posix()
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name != "Validator":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                        items.append((rel, node.name, item.name))
    return items


def _install_shims_if_needed():
    try:
        import duckdb  # noqa: F401
        import pandas  # noqa: F401
        return
    except Exception:
        pass
    import types

    class DummyDF:
        columns = []
        def rename(self, *a, **kw):
            pass

    class DummyResult:
        def fetchdf(self):
            return DummyDF()
        def df(self):
            return DummyDF()

    class DummyConn:
        def execute(self, *a, **kw):
            return DummyResult()
        def close(self):
            pass

    duckdb_mod = types.ModuleType("duckdb")
    duckdb_mod.connect = lambda *a, **kw: DummyConn()
    sys.modules["duckdb"] = duckdb_mod

    pandas_mod = types.ModuleType("pandas")
    pandas_mod.DataFrame = lambda *a, **kw: DummyDF()
    testing_mod = types.ModuleType("pandas.testing")
    testing_mod.assert_frame_equal = lambda *a, **kw: None
    pandas_mod.testing = testing_mod
    sys.modules["pandas"] = pandas_mod
    sys.modules["pandas.testing"] = testing_mod


def _preimport_all_dialects():
    import sqlglot.dialects as dialects_pkg
    for dname in getattr(dialects_pkg, "DIALECTS", ()):
        getattr(dialects_pkg, dname)


def _is_const_none_or_bool(n: ast.AST) -> bool:
    return isinstance(n, ast.Constant) and (n.value is None or isinstance(n.value, bool))


def _is_const_none(n: ast.AST) -> bool:
    return isinstance(n, ast.Constant) and n.value is None


def _collect_static_targets(repo_root: Path) -> set[str]:
    tok_core = repo_root / "sqlglot" / "tokenizer_core.py"
    tok_names: set[str] = set()
    if tok_core.exists():
        tree = ast.parse(tok_core.read_text(encoding="utf-8"))
        for n in tree.body:
            if isinstance(n, ast.ClassDef) and n.name == "TokenType":
                for item in n.body:
                    if isinstance(item, ast.Assign) and item.targets and isinstance(item.targets[0], ast.Name):
                        tok_names.add(item.targets[0].id)

    attrs: set[str] = set()
    dict_attrs: set[str] = set()
    targets: list[Path] = []
    for folder in ("dialects", "parsers", "generators"):
        pdir = repo_root / "sqlglot" / folder
        if pdir.is_dir():
            targets.extend(sorted(pdir.glob("*.py")))
    for fname in ("tokens.py", "parser.py", "generator.py"):
        fp = repo_root / "sqlglot" / fname
        if fp.exists():
            targets.append(fp)

    for p in targets:
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef):
                for item in n.body:
                    if isinstance(item, ast.Assign) and item.targets and isinstance(item.targets[0], ast.Name):
                        nm = item.targets[0].id
                        if nm.isupper() and not nm.startswith("_"):
                            if isinstance(item.value, ast.Dict):
                                dict_attrs.add(nm)
                            else:
                                attrs.add(nm)
                    elif isinstance(item, ast.ClassDef):
                        for sub in item.body:
                            if isinstance(sub, ast.Assign) and sub.targets and isinstance(sub.targets[0], ast.Name):
                                nm = sub.targets[0].id
                                if nm.isupper() and not nm.startswith("_"):
                                    if isinstance(sub.value, ast.Dict):
                                        dict_attrs.add(nm)
                                    else:
                                        attrs.add(nm)
    return attrs - dict_attrs - DERIVED_SOURCE_ATTRS - tok_names - {"NUMERIC_ESCAPES", "KEYWORDS", "SINGLE_TOKENS"}


class _WorkerASTTransformer(ast.NodeTransformer):
    def __init__(self, rel: str, sites: list, scalar_attrs: set[str]):
        self.rel = rel
        self.sites = sites
        self.scalar_attrs = scalar_attrs

    def visit_Attribute(self, node: ast.Attribute):
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Load) and node.attr in self.scalar_attrs:
            call = ast.Call(
                func=ast.Name(id="_blast_getattr", ctx=ast.Load()),
                args=[node.value, ast.Constant(node.attr)],
                keywords=[],
            )
            return ast.copy_location(call, node)
        return node

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) == 2
            and not node.keywords
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
            and node.args[1].value in self.scalar_attrs
        ):
            call = ast.Call(
                func=ast.Name(id="_blast_getattr", ctx=ast.Load()),
                args=[node.args[0], node.args[1]],
                keywords=[],
            )
            return ast.copy_location(call, node)
        return node

    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        if len(node.ops) == 1:
            op = node.ops[0]
            op_code = type(op).__name__
            if isinstance(op, (ast.Is, ast.IsNot)):
                sid = len(self.sites)
                self.sites.append((self.rel, node.lineno, node.col_offset, op_code))
                if _is_const_none(node.comparators[0]):
                    fn = "_blast_cmp_IsNone" if isinstance(op, ast.Is) else "_blast_cmp_IsNotNone"
                    call = ast.Call(
                        func=ast.Name(id=fn, ctx=ast.Load()),
                        args=[ast.Constant(sid), node.left],
                        keywords=[],
                    )
                    return ast.copy_location(call, node)
                if _is_const_none(node.left):
                    fn = "_blast_cmp_IsNone" if isinstance(op, ast.Is) else "_blast_cmp_IsNotNone"
                    call = ast.Call(
                        func=ast.Name(id=fn, ctx=ast.Load()),
                        args=[ast.Constant(sid), node.comparators[0]],
                        keywords=[],
                    )
                    return ast.copy_location(call, node)
                call = ast.Call(
                    func=ast.Name(id=f"_blast_cmp_{op_code}", ctx=ast.Load()),
                    args=[ast.Constant(sid), node.left, node.comparators[0]],
                    keywords=[],
                )
                return ast.copy_location(call, node)
            if isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                sid = len(self.sites)
                self.sites.append((self.rel, node.lineno, node.col_offset, op_code))
                call = ast.Call(
                    func=ast.Name(id=f"_blast_cmp_{op_code}", ctx=ast.Load()),
                    args=[ast.Constant(sid), node.left, node.comparators[0]],
                    keywords=[],
                )
                return ast.copy_location(call, node)
            if isinstance(op, (ast.Eq, ast.NotEq)):
                if _is_const_none_or_bool(node.left) or _is_const_none_or_bool(node.comparators[0]):
                    sid = len(self.sites)
                    self.sites.append((self.rel, node.lineno, node.col_offset, op_code))
                    call = ast.Call(
                        func=ast.Name(id=f"_blast_cmp_{op_code}", ctx=ast.Load()),
                        args=[ast.Constant(sid), node.left, node.comparators[0]],
                        keywords=[],
                    )
                    return ast.copy_location(call, node)
        return node


def _trace_worker(args):
    repo_root_str, file_list, scalar_attrs = args
    repo_root = Path(repo_root_str)
    os.environ["SKIP_INTEGRATION"] = "0"
    sys.path.insert(0, str(repo_root))
    _install_shims_if_needed()
    concurrent.futures.ProcessPoolExecutor = _SyncPool

    sites: list[tuple[str, int, int, str]] = []
    cur_bad: set[int] = set()
    cls_bad: set[int] = set()
    cur_dict_hits: set[tuple[str, str, str]] = set()
    cls_dict_hits: set[tuple[str, str, str]] = set()
    cur_attr_hits: set[tuple[str, str]] = set()
    cls_attr_hits: set[tuple[str, str]] = set()
    active_test = False
    active_cmp = False
    in_setup = False

    def rec_dict(tag: tuple[str, str], ktag: str):
        if active_test or (active_cmp and not in_setup):
            cur_dict_hits.add((tag[0], tag[1], ktag))
        elif in_setup:
            cls_dict_hits.add((tag[0], tag[1], ktag))

    def _cmp_is_none(sid, l):
        if l is None:
            return True
        if active_cmp:
            dst = cur_bad if not in_setup else cls_bad
            if sid not in dst:
                t = type(l)
                if t.__eq__ is not object.__eq__ and t not in _SAFE_NONE_EQ_TYPES:
                    try:
                        eq = (l == None)
                        if type(eq) is not bool or eq:
                            dst.add(sid)
                    except Exception:
                        dst.add(sid)
        return False

    def _cmp_is_not_none(sid, l):
        if l is None:
            return False
        if active_cmp:
            dst = cur_bad if not in_setup else cls_bad
            if sid not in dst:
                t = type(l)
                if t.__eq__ is not object.__eq__ and t not in _SAFE_NONE_EQ_TYPES:
                    try:
                        neq = (l != None)
                        if type(neq) is not bool or not neq:
                            dst.add(sid)
                    except Exception:
                        dst.add(sid)
        return True

    def _cmp_is(sid, l, r):
        res = l is r
        if active_cmp:
            dst = cur_bad if not in_setup else cls_bad
            if sid not in dst:
                try:
                    alt = l == r
                    if type(alt) is not bool or alt is not res:
                        dst.add(sid)
                except Exception:
                    dst.add(sid)
        return res

    def _cmp_is_not(sid, l, r):
        res = l is not r
        if active_cmp:
            dst = cur_bad if not in_setup else cls_bad
            if sid not in dst:
                try:
                    alt = l != r
                    if type(alt) is not bool or alt is not res:
                        dst.add(sid)
                except Exception:
                    dst.add(sid)
        return res

    def _cmp_eq(sid, l, r):
        res = l == r
        if active_cmp:
            dst = cur_bad if not in_setup else cls_bad
            if sid not in dst:
                if (l is r) is not res or type(res) is not bool:
                    dst.add(sid)
        return res

    def _cmp_neq(sid, l, r):
        res = l != r
        if active_cmp:
            dst = cur_bad if not in_setup else cls_bad
            if sid not in dst:
                if (l is not r) is not res or type(res) is not bool:
                    dst.add(sid)
        return res

    def _cmp_lt(sid, l, r):
        res = l < r
        if active_cmp:
            dst = cur_bad if not in_setup else cls_bad
            if sid not in dst:
                try:
                    alt = l <= r
                    if type(alt) is not bool or alt is not res:
                        dst.add(sid)
                except Exception:
                    dst.add(sid)
        return res

    def _cmp_lte(sid, l, r):
        res = l <= r
        if active_cmp:
            dst = cur_bad if not in_setup else cls_bad
            if sid not in dst:
                try:
                    alt = l < r
                    if type(alt) is not bool or alt is not res:
                        dst.add(sid)
                except Exception:
                    dst.add(sid)
        return res

    def _cmp_gt(sid, l, r):
        res = l > r
        if active_cmp:
            dst = cur_bad if not in_setup else cls_bad
            if sid not in dst:
                try:
                    alt = l >= r
                    if type(alt) is not bool or alt is not res:
                        dst.add(sid)
                except Exception:
                    dst.add(sid)
        return res

    def _cmp_gte(sid, l, r):
        res = l >= r
        if active_cmp:
            dst = cur_bad if not in_setup else cls_bad
            if sid not in dst:
                try:
                    alt = l > r
                    if type(alt) is not bool or alt is not res:
                        dst.add(sid)
                except Exception:
                    dst.add(sid)
        return res

    def _first_sqlglot_base(tp):
        for base in tp.__mro__[1:]:
            if base is object:
                break
            if base.__module__.startswith("sqlglot."):
                return base
        return None

    def _tracked_getattr(obj, name):
        val = getattr(obj, name)
        if active_cmp:
            dst = cls_attr_hits if in_setup else cur_attr_hits
            if isinstance(obj, type):
                dst.add((obj.__qualname__, name))
                if obj.__module__.startswith("tests."):
                    b = _first_sqlglot_base(obj)
                    if b is not None:
                        dst.add((b.__qualname__, name))
            else:
                d = getattr(obj, "dialect", None)
                if d is not None:
                    dt = type(d)
                    dst.add((dt.__name__, name))
                    if dt.__module__.startswith("tests."):
                        b = _first_sqlglot_base(dt)
                        if b is not None:
                            dst.add((b.__name__, name))
                else:
                    ot = type(obj)
                    dst.add((ot.__qualname__, name))
                    if ot.__module__.startswith("tests."):
                        b = _first_sqlglot_base(ot)
                        if b is not None:
                            dst.add((b.__qualname__, name))
        return val

    builtins.__dict__["_blast_cmp_IsNone"] = _cmp_is_none
    builtins.__dict__["_blast_cmp_IsNotNone"] = _cmp_is_not_none
    builtins.__dict__["_blast_cmp_Is"] = _cmp_is
    builtins.__dict__["_blast_cmp_IsNot"] = _cmp_is_not
    builtins.__dict__["_blast_cmp_Eq"] = _cmp_eq
    builtins.__dict__["_blast_cmp_NotEq"] = _cmp_neq
    builtins.__dict__["_blast_cmp_Lt"] = _cmp_lt
    builtins.__dict__["_blast_cmp_LtE"] = _cmp_lte
    builtins.__dict__["_blast_cmp_Gt"] = _cmp_gt
    builtins.__dict__["_blast_cmp_GtE"] = _cmp_gte
    builtins.__dict__["_blast_getattr"] = _tracked_getattr

    class _BlastFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if not (fullname == "sqlglot" or fullname.startswith("sqlglot.")):
                return None
            parts = fullname.split(".")
            cand_pkg = repo_root.joinpath(*parts, "__init__.py")
            cand_mod = repo_root.joinpath(*parts[:-1], parts[-1] + ".py")
            if cand_pkg.exists():
                src_path = cand_pkg
                sub = [str(cand_pkg.parent)]
            elif cand_mod.exists():
                src_path = cand_mod
                sub = None
            else:
                return None
            rel = src_path.relative_to(repo_root).as_posix()
            return importlib.util.spec_from_file_location(
                fullname, str(src_path), loader=_BlastLoader(src_path, rel), submodule_search_locations=sub
            )

    class _BlastLoader(importlib.abc.Loader):
        def __init__(self, path: Path, rel: str):
            self.path = path
            self.rel = rel

        def create_module(self, spec):
            return None

        def exec_module(self, module):
            src = self.path.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=str(self.path))
            tree = _WorkerASTTransformer(self.rel, sites, scalar_attrs).visit(tree)
            ast.fix_missing_locations(tree)
            exec(compile(tree, str(self.path), "exec"), module.__dict__)

    finder = _BlastFinder()
    sys.meta_path.insert(0, finder)

    # Pre-import all dialects through BlastLoader before tracing so lazy imports don't blame unrelated tests
    _preimport_all_dialects()

    sqlglot_root = str((repo_root / "sqlglot").resolve())
    fixtures_root = str((repo_root / "tests" / "fixtures").resolve())

    from sqlglot.dialects.dialect import Dialect
    from sqlglot.tokens import Tokenizer
    from sqlglot.parser import Parser
    from sqlglot.generator import Generator

    # Wrap eligible uppercase class dicts AFTER pre-importing dialects
    from sqlglot.dialects.dialect import Dialects
    wrapped_ids: dict[int, TrackedDict] = {}
    all_classes: set[type] = set()
    w_seen_dialects: set[type] = set(Dialect.classes.values())
    for d in Dialects:
        w_seen_dialects.add(type(Dialect.get_or_raise(d)))
    w_stack = list(w_seen_dialects)
    while w_stack:
        c = w_stack.pop()
        for sub in c.__subclasses__():
            if sub not in w_seen_dialects:
                w_seen_dialects.add(sub)
                w_stack.append(sub)
    for d_cls in w_seen_dialects:
        try:
            inst = d_cls()
            p_cls = type(inst.parser())
            g_cls = type(inst.generator())
            t_cls = type(inst.tokenizer())
        except Exception:
            p_cls = g_cls = t_cls = None
        for root in (
            d_cls,
            p_cls,
            g_cls,
            t_cls,
            getattr(d_cls, "Parser", None),
            getattr(d_cls, "Generator", None),
            getattr(d_cls, "Tokenizer", None),
            getattr(d_cls, "JSONPathTokenizer", None),
            getattr(d_cls, "jsonpath_tokenizer_class", None),
        ):
            if isinstance(root, type):
                for mcls in root.__mro__:
                    if mcls is not object:
                        all_classes.add(mcls)
    for cls in sorted(all_classes, key=lambda c: c.__qualname__):
        for k, v in list(cls.__dict__.items()):
            if not k.startswith("_") and k.isupper() and type(v) is dict and k not in DERIVED_SOURCE_ATTRS:
                td = wrapped_ids.get(id(v))
                if td is None:
                    td = TrackedDict(v, (cls.__qualname__, k))
                    wrapped_ids[id(v)] = td
                setattr(cls, k, td)

    TrackedDict._rec_fn = rec_dict

    current_dialects: set[str] = set()
    class_dialects: set[str] = set()
    current_tok: set[str] = set()
    class_tok: set[str] = set()
    current_parse: set[str] = set()
    class_parse: set[str] = set()
    current_gen_init: set[str] = set()
    class_gen_init: set[str] = set()
    current_gen_sql: set[str] = set()
    class_gen_sql: set[str] = set()
    active_test = False

    def rec_set(target_cur, target_cls, name):
        if active_test:
            target_cur.add(name)
        else:
            target_cls.add(name)

    orig_get_or_raise = Dialect.get_or_raise.__func__

    @classmethod
    def tracked_get_or_raise(cls, dialect):
        res = orig_get_or_raise(cls, dialect)
        rec_set(current_dialects, class_dialects, type(res).__name__)
        return res

    Dialect.get_or_raise = tracked_get_or_raise

    orig_dialect_init = Dialect.__init__

    def tracked_dialect_init(self, *a, **kw):
        orig_dialect_init(self, *a, **kw)
        rec_set(current_dialects, class_dialects, type(self).__name__)

    Dialect.__init__ = tracked_dialect_init

    orig_tok_init = Tokenizer.__init__

    def tracked_tok_init(self, *a, **kw):
        orig_tok_init(self, *a, **kw)
        dn = type(self.dialect).__name__
        rec_set(current_dialects, class_dialects, dn)
        rec_set(current_tok, class_tok, dn)

    Tokenizer.__init__ = tracked_tok_init

    orig_parse_init = Parser.__init__

    def tracked_parse_init(self, *a, **kw):
        orig_parse_init(self, *a, **kw)
        dn = type(self.dialect).__name__
        rec_set(current_dialects, class_dialects, dn)
        rec_set(current_parse, class_parse, dn)

    Parser.__init__ = tracked_parse_init

    orig_gen_init = Generator.__init__

    def tracked_gen_init(self, *a, **kw):
        orig_gen_init(self, *a, **kw)
        dn = type(self.dialect).__name__
        rec_set(current_dialects, class_dialects, dn)
        rec_set(current_gen_init, class_gen_init, dn)

    Generator.__init__ = tracked_gen_init

    orig_gen_sql = Generator.sql

    def tracked_gen_sql(self, *a, **kw):
        rec_set(current_gen_sql, class_gen_sql, type(self.dialect).__name__)
        return orig_gen_sql(self, *a, **kw)

    Generator.sql = tracked_gen_sql

    # Hook open / gzip.open to track fixtures opened per test and per class
    orig_open = builtins.open
    orig_gzip_open = gzip.open
    current_fixtures: set[str] = set()
    class_fixtures: set[str] = set()

    def record_fixture(p):
        try:
            ap = os.path.abspath(os.fspath(p))
        except Exception:
            return
        if ap.startswith(fixtures_root + os.sep) or ap.startswith(fixtures_root + "/"):
            rel = Path(ap).relative_to(repo_root).as_posix()
            if active_test:
                current_fixtures.add(rel)
            else:
                class_fixtures.add(rel)

    def wrapped_open(file, *a, **kw):
        record_fixture(file)
        return orig_open(file, *a, **kw)

    def wrapped_gzip_open(filename, *a, **kw):
        record_fixture(filename)
        return orig_gzip_open(filename, *a, **kw)

    builtins.open = wrapped_open
    gzip.open = wrapped_gzip_open

    current_lines: set[tuple[str, int]] = set()
    code_rel_cache: dict = {}

    def get_rel_for_code(filename: str) -> str | None:
        res = code_rel_cache.get(filename)
        if res is not None:
            return res if res != "" else None
        try:
            ap = os.path.abspath(filename)
        except Exception:
            code_rel_cache[filename] = ""
            return None
        if (ap.startswith(sqlglot_root + os.sep) or ap.startswith(sqlglot_root + "/")) and ap.endswith(".py"):
            rel = Path(ap).relative_to(repo_root).as_posix()
            code_rel_cache[filename] = rel
            return rel
        code_rel_cache[filename] = ""
        return None

    use_monitoring = hasattr(sys, "monitoring")
    if use_monitoring:
        mon = sys.monitoring
        TOOL_ID = mon.COVERAGE_ID
        mon.use_tool_id(TOOL_ID, "blast_cov")

        def on_py_start(code, offset):
            rel = get_rel_for_code(code.co_filename)
            if rel is not None:
                mon.set_local_events(TOOL_ID, code, mon.events.LINE)
            return mon.DISABLE

        def on_line(code, lineno):
            if active_test:
                rel = get_rel_for_code(code.co_filename)
                if rel is not None:
                    current_lines.add((rel, lineno))
            return mon.DISABLE

        mon.register_callback(TOOL_ID, mon.events.PY_START, on_py_start)
        mon.register_callback(TOOL_ID, mon.events.LINE, on_line)
        mon.set_events(TOOL_ID, mon.events.PY_START)
    else:
        trace_hits = 0

        def trace_lines(frame, event, arg):
            nonlocal trace_hits
            if event == "line" and active_test:
                trace_hits += 1
                if (trace_hits & 4095) == 0 and time.monotonic() > trace_deadline:
                    sys.settrace(None)
                    return None
                rel = get_rel_for_code(frame.f_code.co_filename)
                if rel is not None:
                    current_lines.add((rel, frame.f_lineno))
            return trace_lines

        def trace_fn(frame, event, arg):
            if event == "call" and active_test:
                rel = get_rel_for_code(frame.f_code.co_filename)
                if rel is not None:
                    return trace_lines
            return None

    import time
    results = []
    for file_path, cls_items in file_list:
        mod_name = file_path[:-3].replace("/", ".")
        mod = importlib.import_module(mod_name)
        if hasattr(mod, "ProcessPoolExecutor"):
            mod.ProcessPoolExecutor = _SyncPool

        by_cls = defaultdict(list)
        for c, m in cls_items:
            by_cls[c].append(m)

        for cls_name, meths in by_cls.items():
            cls = getattr(mod, cls_name)
            if not (isinstance(cls, type) and issubclass(cls, unittest.TestCase)):
                continue
            class_fixtures.clear()
            class_dialects.clear()
            class_tok.clear()
            class_parse.clear()
            class_gen_init.clear()
            class_gen_sql.clear()
            cls_bad.clear()
            cls_dict_hits.clear()
            cls_attr_hits.clear()
            active_test = False
            in_setup = True
            active_cmp = True
            if hasattr(cls, "setUpClass"):
                try:
                    cls.setUpClass()
                except Exception:
                    pass
            in_setup = False
            active_cmp = False
            saved_class_fixtures = set(class_fixtures)
            saved_class_dialects = set(class_dialects)
            saved_class_tok = set(class_tok)
            saved_class_parse = set(class_parse)
            saved_class_gen_init = set(class_gen_init)
            saved_class_gen_sql = set(class_gen_sql)
            saved_bad = set(cls_bad)
            saved_dict_hits = set(cls_dict_hits)
            saved_attr_hits = set(cls_attr_hits)

            for meth_name in meths:
                nodeid = f"{file_path}::{cls_name}::{meth_name}"
                current_lines.clear()
                current_fixtures.clear()
                current_dialects.clear()
                current_tok.clear()
                current_parse.clear()
                current_gen_init.clear()
                current_gen_sql.clear()
                cur_bad.clear()
                cur_dict_hits.clear()
                cur_attr_hits.clear()
                inst = cls(meth_name)
                active_test = True
                active_cmp = True
                trace_hits = 0
                trace_deadline = time.monotonic() + 15.0
                if use_monitoring:
                    sys.monitoring.restart_events()
                else:
                    sys.settrace(trace_fn)
                try:
                    inst.setUp()
                    getattr(inst, meth_name)()
                    inst.tearDown()
                except Exception:
                    pass
                finally:
                    active_test = False
                    if not use_monitoring:
                        sys.settrace(None)
                    try:
                        inst.doCleanups()
                    except Exception:
                        pass
                    active_cmp = False
                all_fx = set(current_fixtures)
                for fx in saved_class_fixtures:
                    if cls_name == "TestExecutor":
                        if "tpc-h" in fx and "tpch" not in meth_name:
                            continue
                        if "tpc-ds" in fx and "tpcds" not in meth_name:
                            continue
                    all_fx.add(fx)
                results.append((
                    nodeid,
                    [f"{r}:{ln}" for r, ln in current_lines],
                    sorted(all_fx),
                    sorted(current_dialects | saved_class_dialects),
                    sorted(current_tok | saved_class_tok),
                    sorted(current_parse | saved_class_parse),
                    sorted(current_gen_init | saved_class_gen_init),
                    sorted(current_gen_sql | saved_class_gen_sql),
                    [f"{r}:{ln}:{col}:{op}" for (r, ln, col, op) in (sites[s] for s in (cur_bad | saved_bad))],
                    [f"{c}|{a}|{k}" for (c, a, k) in (cur_dict_hits | saved_dict_hits)],
                    [f"{d}|{a}" for (d, a) in (cur_attr_hits | saved_attr_hits)],
                ))

            if hasattr(cls, "tearDownClass"):
                try:
                    cls.tearDownClass()
                except Exception:
                    pass
            if hasattr(cls, "doClassCleanups"):
                try:
                    cls.doClassCleanups()
                except Exception:
                    pass

    TrackedDict._rec_fn = None
    if finder in sys.meta_path:
        sys.meta_path.remove(finder)
    builtins.open = orig_open
    gzip.open = orig_gzip_open
    Dialect.get_or_raise = orig_get_or_raise
    Dialect.__init__ = orig_dialect_init
    Tokenizer.__init__ = orig_tok_init
    Parser.__init__ = orig_parse_init
    Generator.__init__ = orig_gen_init
    Generator.sql = orig_gen_sql
    return results


def build_all(repo_root: Path, out_path: Path, workers: int | None = None) -> None:
    items = _discover_unittest_items(repo_root)
    scalar_attrs = _collect_static_targets(repo_root)
    n_workers = workers or min(8, os.cpu_count() or 4)
    buckets_map = [defaultdict(list) for _ in range(n_workers)]
    for i, (rel, cls_name, meth_name) in enumerate(items):
        buckets_map[i % n_workers][rel].append((cls_name, meth_name))
    buckets = [sorted(b.items()) for b in buckets_map if b]

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        worker_outputs = pool.map(_trace_worker, [(str(repo_root), b, scalar_attrs) for b in buckets if b])

    nodeids = []
    line_to_idxs = defaultdict(set)
    fixture_to_idxs = defaultdict(set)
    dialect_to_idxs = defaultdict(set)
    tok_to_idxs = defaultdict(set)
    parse_to_idxs = defaultdict(set)
    gen_init_to_idxs = defaultdict(set)
    gen_sql_to_idxs = defaultdict(set)
    cmp_diff_to_idxs = defaultdict(set)
    dict_key_to_idxs = defaultdict(set)
    scalar_attr_to_idxs = defaultdict(set)

    for wout in worker_outputs:
        for nodeid, lines, fxs, dls, toks, parses, ginits, gsqls, cmp_bads, dict_hits, attr_hits in wout:
            idx = len(nodeids)
            nodeids.append(nodeid)
            for loc in lines:
                line_to_idxs[loc].add(idx)
            for fx in fxs:
                fixture_to_idxs[fx].add(idx)
            for dl in dls:
                dialect_to_idxs[dl].add(idx)
            for t in toks:
                tok_to_idxs[t].add(idx)
            for p in parses:
                parse_to_idxs[p].add(idx)
            for gi in ginits:
                gen_init_to_idxs[gi].add(idx)
            for gs in gsqls:
                gen_sql_to_idxs[gs].add(idx)
            for cb in cmp_bads:
                cmp_diff_to_idxs[cb].add(idx)
            for dh in dict_hits:
                dict_key_to_idxs[dh].add(idx)
            for ah in attr_hits:
                scalar_attr_to_idxs[ah].add(idx)

    # Static AST + inheritance analysis
    sys.path.insert(0, str(repo_root))
    _preimport_all_dialects()
    from sqlglot.dialects.dialect import Dialect, Dialects
    from sqlglot.tokens import Tokenizer
    from sqlglot.parser import Parser
    from sqlglot.generator import Generator

    dialect_subclasses = defaultdict(set)
    parser_subclasses = defaultdict(set)
    generator_subclasses = defaultdict(set)
    tokenizer_subclasses = defaultdict(set)

    seen_dialect_cls = set(Dialect.classes.values())
    for d in Dialects:
        seen_dialect_cls.add(type(Dialect.get_or_raise(d)))
    stack = list(seen_dialect_cls)
    while stack:
        c = stack.pop()
        for sub in c.__subclasses__():
            if sub not in seen_dialect_cls:
                seen_dialect_cls.add(sub)
                stack.append(sub)

    all_tracked_classes: set[type] = set()
    for d_cls in seen_dialect_cls:
        d_name = d_cls.__name__
        try:
            inst = d_cls()
            p_cls = type(inst.parser())
            g_cls = type(inst.generator())
            t_cls = type(inst.tokenizer())
            jt_cls = getattr(d_cls, "jsonpath_tokenizer_class", None)
        except Exception:
            p_cls = None
            g_cls = None
            t_cls = None
            jt_cls = None
        for root in (d_cls, p_cls, g_cls, t_cls, jt_cls):
            if isinstance(root, type):
                for base in root.__mro__:
                    if base is not object:
                        all_tracked_classes.add(base)
        for base in d_cls.__mro__:
            if issubclass(base, Dialect):
                dialect_subclasses[base.__name__].add(d_name)
        if p_cls is not None:
            for base in p_cls.__mro__:
                if issubclass(base, Parser):
                    parser_subclasses[base.__name__].add(d_name)
        if g_cls is not None:
            for base in g_cls.__mro__:
                if issubclass(base, Generator):
                    generator_subclasses[base.__name__].add(d_name)
        for tc in (t_cls, jt_cls):
            if tc is not None:
                for base in tc.__mro__:
                    if isinstance(base, type) and issubclass(base, Tokenizer):
                        tokenizer_subclasses[base.__qualname__].add(d_name)
                        if base.__name__ not in ("Tokenizer", "JSONPathTokenizer"):
                            tokenizer_subclasses[base.__name__].add(d_name)

    cls_sub_qualnames: dict[str, set[str]] = defaultdict(set)
    tracked_dicts: set[tuple[str, str]] = set()
    for cls in all_tracked_classes:
        cq = cls.__qualname__
        for base in cls.__mro__:
            if base is not object:
                cls_sub_qualnames[base.__qualname__].add(cq)
        for k, v in cls.__dict__.items():
            if not k.startswith("_") and k.isupper() and isinstance(v, dict) and k not in DERIVED_SOURCE_ATTRS:
                tracked_dicts.add((cq, k))

    mod_to_file = {}
    file_data = {}
    attr_readers = defaultdict(list)
    name_readers = defaultdict(list)
    mod_attr_readers = defaultdict(set)
    mod_name_readers = defaultdict(set)
    defined_symbols: set[str] = set()
    tracked_dict_attrs = {a for _, a in tracked_dicts}
    test_dict_readers: dict[str, list[str]] = defaultdict(list)

    all_py = list(sorted((repo_root / "sqlglot").rglob("*.py"))) + list(sorted((repo_root / "tests").rglob("*.py")))
    for p in all_py:
        rel = p.relative_to(repo_root).as_posix()
        src = p.read_text(encoding="utf-8")
        info = analyze_python_file(rel, src)
        file_data[rel] = info
        mod_to_file[info["module"]] = rel
        if rel.startswith("sqlglot/"):
            for li in info["line_info"].values():
                if "class_attr" in li:
                    cn, an = li["class_attr"]
                    if cn:
                        defined_symbols.add(cn)
                        defined_symbols.add(cn.split(".")[-1])
                    if an:
                        defined_symbols.add(an)
                if "module_var" in li and li["module_var"]:
                    defined_symbols.add(li["module_var"])
            for attr, lns in info["attr_reads"].items():
                for ln in lns:
                    attr_readers[attr].append(f"{rel}:{ln}")
            for nm, lns in info["name_reads"].items():
                for ln in lns:
                    name_readers[nm].append(f"{rel}:{ln}")
            for attr in info["mod_attr_reads"]:
                mod_attr_readers[attr].add(rel)
            for nm in info["mod_name_reads"]:
                mod_name_readers[nm].add(rel)
        else:
            for attr, lns in info["attr_reads"].items():
                if attr in scalar_attrs:
                    for ln in lns:
                        attr_readers[attr].append(f"{rel}:{ln}")
                if attr in tracked_dict_attrs:
                    for ln in lns:
                        test_dict_readers[attr].append(f"{rel}:{ln}")

    # Transitive dependency mapping for every python file (sqlglot + tests)
    # reverse_importers[target_rel] = all rels that transitively import target_rel
    direct_deps: dict[str, set[str]] = {}
    for rel, info in file_data.items():
        deps = set()
        for imp in info["imports"]:
            if imp in mod_to_file:
                deps.add(mod_to_file[imp])
            else:
                # e.g. sqlglot.helper.seq_get -> sqlglot.helper
                parts = imp.split(".")
                for k in range(len(parts) - 1, 0, -1):
                    cand = ".".join(parts[:k])
                    if cand in mod_to_file:
                        deps.add(mod_to_file[cand])
                        break
        direct_deps[rel] = deps

    file_dep_closure: dict[str, set[str]] = {}
    for rel in file_data:
        seen = set(direct_deps.get(rel, ()))
        st = list(seen)
        while st:
            cur = st.pop()
            for dep in direct_deps.get(cur, ()):
                if dep not in seen:
                    seen.add(dep)
                    st.append(dep)
        file_dep_closure[rel] = seen

    lines_by_file: dict[str, dict[int, set[int]]] = defaultdict(dict)
    file_cov_by_file: dict[str, set[int]] = defaultdict(set)
    for key, idxs in line_to_idxs.items():
        rel, ln_str = key.rsplit(":", 1)
        ln = int(ln_str)
        lines_by_file[rel][ln] = idxs
        file_cov_by_file[rel].update(idxs)

    if out_path.suffix == ".pkl":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        meta_dir = out_path.parent / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        file_ids = {}
        ast_hashes = {}
        test_methods_map = {}
        for fid, (rel, info) in enumerate(file_data.items()):
            file_ids[rel] = fid
            ast_hashes[rel] = info["ast_hash"]
            if info["test_methods"]:
                test_methods_map[rel] = info["test_methods"]
            with open(meta_dir / f"{fid}.pkl", "wb") as mf:
                pickle.dump(
                    {
                        "line_to_idxs": lines_by_file.get(rel, {}),
                        "file_cov_idxs": file_cov_by_file.get(rel, set()),
                        "line_info": info["line_info"],
                        "stmt_spans": info["stmt_spans"],
                        "str_lines": info["str_lines"],
                    },
                    mf,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

        filtered_attr = {k: v for k, v in attr_readers.items() if k in defined_symbols}
        filtered_name = {k: v for k, v in name_readers.items() if k in defined_symbols}
        with open(out_path.parent / "readers.pkl", "wb") as rf:
            pickle.dump(
                {"attr_readers": filtered_attr, "name_readers": filtered_name},
                rf,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        with open(out_path.parent / "dict_keys.pkl", "wb") as dk:
            pickle.dump(dict(dict_key_to_idxs), dk, protocol=pickle.HIGHEST_PROTOCOL)

        index_payload = {
            "split": True,
            "nodeids": nodeids,
            "file_ids": file_ids,
            "ast_hashes": ast_hashes,
            "test_methods": test_methods_map,
            "fixture_to_idxs": dict(fixture_to_idxs),
            "dialect_to_idxs": dict(dialect_to_idxs),
            "tok_to_idxs": dict(tok_to_idxs),
            "parse_to_idxs": dict(parse_to_idxs),
            "gen_init_to_idxs": dict(gen_init_to_idxs),
            "gen_sql_to_idxs": dict(gen_sql_to_idxs),
            "dialect_subclasses": dict(dialect_subclasses),
            "parser_subclasses": dict(parser_subclasses),
            "generator_subclasses": dict(generator_subclasses),
            "tokenizer_subclasses": dict(tokenizer_subclasses),
            "cls_sub_qualnames": {k: sorted(v) for k, v in cls_sub_qualnames.items()},
            "tracked_dicts": sorted(tracked_dicts),
            "test_dict_readers": dict(test_dict_readers),
            "scalar_attrs": sorted(scalar_attrs),
            "scalar_attr_to_idxs": dict(scalar_attr_to_idxs),
            "cmp_diff_to_idxs": dict(cmp_diff_to_idxs),
            "mod_attr_readers": {k: sorted(v) for k, v in mod_attr_readers.items()},
            "mod_name_readers": {k: sorted(v) for k, v in mod_name_readers.items()},
            "file_deps": file_dep_closure,
        }
        with open(out_path, "wb") as f:
            pickle.dump(index_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        total_mb = (
            out_path.stat().st_size
            + (out_path.parent / "readers.pkl").stat().st_size
            + (out_path.parent / "dict_keys.pkl").stat().st_size
            + sum(p.stat().st_size for p in meta_dir.iterdir())
        ) / 1e6
        print(
            f"Built split {out_path} (index={out_path.stat().st_size / 1e6:.2f} MB, total={total_mb:.2f} MB, cmp_diffs={len(cmp_diff_to_idxs)}, dict_keys={len(dict_key_to_idxs)}, scalar_attrs={len(scalar_attr_to_idxs)}) for {len(nodeids)} tests",
            flush=True,
        )
        return

    compact_file_data = {}
    for rel, info in file_data.items():
        compact_file_data[rel] = {
            "ast_hash": info["ast_hash"],
            "line_info": info["line_info"],
            "stmt_spans": info["stmt_spans"],
            "test_methods": info["test_methods"],
        }

    payload = {
        "nodeids": nodeids,
        "line_to_idxs": dict(line_to_idxs),
        "fixture_to_idxs": dict(fixture_to_idxs),
        "dialect_to_idxs": dict(dialect_to_idxs),
        "tok_to_idxs": dict(tok_to_idxs),
        "parse_to_idxs": dict(parse_to_idxs),
        "gen_init_to_idxs": dict(gen_init_to_idxs),
        "gen_sql_to_idxs": dict(gen_sql_to_idxs),
        "dialect_subclasses": dict(dialect_subclasses),
        "parser_subclasses": dict(parser_subclasses),
        "generator_subclasses": dict(generator_subclasses),
        "tokenizer_subclasses": dict(tokenizer_subclasses),
        "file_data": compact_file_data,
        "attr_readers": dict(attr_readers),
        "name_readers": dict(name_readers),
        "mod_attr_readers": {k: sorted(v) for k, v in mod_attr_readers.items()},
        "mod_name_readers": {k: sorted(v) for k, v in mod_name_readers.items()},
        "file_deps": file_dep_closure,
    }
    json_payload = {
        k: (
            {kk: sorted(vv) for kk, vv in v.items()}
            if isinstance(v, dict) and v and isinstance(next(iter(v.values())), set)
            else v
        )
        for k, v in payload.items()
    }
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(json_payload, f, separators=(",", ":"))
    print(f"Built {out_path} ({out_path.stat().st_size / 1e6:.2f} MB) for {len(nodeids)} tests", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: build_maps.py <sqlglot_repo_dir> <out_maps.pkl|out_maps.json.gz>", file=sys.stderr)
        sys.exit(2)
    build_all(Path(sys.argv[1]), Path(sys.argv[2]))
