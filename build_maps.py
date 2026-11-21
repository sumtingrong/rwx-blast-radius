"""Build-time coverage, dialect-usage, fixture, and static AST map generator for sqlglot.

Designed for Python 3.13 (uses sys.monitoring + multiprocessing) inside the Linux Docker build,
and also works on Python 3.11+ via sys.settrace fallback.
"""
from __future__ import annotations

import ast
import builtins
import concurrent.futures
import gzip
import importlib
import json
import multiprocessing
import os
import sys
import unittest
from collections import defaultdict
from concurrent.futures import Future
from pathlib import Path


def iter_stmt_spans(node: ast.AST):
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body_start = child.body[0].lineno if child.body else child.end_lineno
            yield (child.lineno, max(child.lineno, body_start - 1), child)
            yield from iter_stmt_spans(child)
        elif isinstance(child, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)):
            if isinstance(child, ast.If):
                yield (child.lineno, child.test.end_lineno or child.lineno, child)
            elif isinstance(child, (ast.For, ast.AsyncFor)):
                yield (child.lineno, child.iter.end_lineno or child.lineno, child)
            elif isinstance(child, ast.While):
                yield (child.lineno, child.test.end_lineno or child.lineno, child)
            else:
                yield (child.lineno, child.lineno, child)
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


def analyze_python_file(rel: str, src: str) -> dict:
    tree = ast.parse(src)
    mod_name = module_name_from_rel(rel)
    is_pkg = rel.endswith("/__init__.py")

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("sqlglot"):
                    imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            for base_mod in resolve_import(mod_name, is_pkg, node.module, node.level):
                if base_mod.startswith("sqlglot"):
                    imports.add(base_mod)
                    for alias in node.names:
                        if alias.name != "*":
                            imports.add(f"{base_mod}.{alias.name}")

    attr_reads = defaultdict(set)
    name_reads = defaultdict(set)

    def walk_reads(node: ast.AST, in_func: bool) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for ch in ast.iter_child_nodes(node):
                walk_reads(ch, True)
            return
        if in_func:
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                attr_reads[node.attr].add(node.lineno)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                name_reads[node.id].add(node.lineno)
        for ch in ast.iter_child_nodes(node):
            walk_reads(ch, in_func)

    walk_reads(tree, False)

    line_info: dict[int, dict] = {}

    def mark_lines(start: int, end: int | None, **kwargs) -> None:
        for ln in range(start, (end or start) + 1):
            line_info.setdefault(ln, {}).update(kwargs)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            mark_lines(node.lineno, node.end_lineno, func=f"{node.name}@{node.lineno}", func_range=[node.lineno, node.end_lineno])

    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            mark_lines(node.lineno, node.end_lineno, is_lambda=True, lambda_range=[node.lineno, node.end_lineno])

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.Assign, ast.AnnAssign)):
                    target_name = None
                    if isinstance(item, ast.Assign) and item.targets and isinstance(item.targets[0], ast.Name):
                        target_name = item.targets[0].id
                    elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        target_name = item.target.id
                    for ln in range(item.lineno, (item.end_lineno or item.lineno) + 1):
                        info = line_info.get(ln, {})
                        if not info.get("is_lambda") and not info.get("func"):
                            line_info.setdefault(ln, {})["class_attr"] = [node.name, target_name]
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            target_name = None
            if isinstance(node, ast.Assign) and node.targets and isinstance(node.targets[0], ast.Name):
                target_name = node.targets[0].id
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
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

    return {
        "module": mod_name,
        "imports": sorted(imports),
        "attr_reads": {k: sorted(v) for k, v in attr_reads.items()},
        "name_reads": {k: sorted(v) for k, v in name_reads.items()},
        "line_info": {str(k): v for k, v in line_info.items()},
        "stmt_spans": {str(k): v for k, v in stmt_spans.items()},
    }


class _SyncPool:
    def __init__(self, *a, initializer=None, initargs=(), **kw):
        if initializer is not None:
            initializer(*initargs)

    def submit(self, fn, *args, **kwargs):
        import pickle
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
            if isinstance(node, ast.ClassDef):
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


def _trace_worker(args):
    repo_root_str, file_list = args
    repo_root = Path(repo_root_str)
    os.environ["SKIP_INTEGRATION"] = "0"
    sys.path.insert(0, str(repo_root))
    _install_shims_if_needed()
    concurrent.futures.ProcessPoolExecutor = _SyncPool

    sqlglot_root = str((repo_root / "sqlglot").resolve())
    fixtures_root = str((repo_root / "tests" / "fixtures").resolve())

    # Hook Dialect.get_or_raise to track dialect classes used per test
    from sqlglot.dialects.dialect import Dialect
    orig_get_or_raise = Dialect.get_or_raise.__func__
    current_dialects: set[str] = set()
    class_dialects: set[str] = set()
    active_test = False

    @classmethod
    def tracked_get_or_raise(cls, dialect):
        res = orig_get_or_raise(cls, dialect)
        name = type(res).__name__
        if active_test:
            current_dialects.add(name)
        else:
            class_dialects.add(name)
        return res

    Dialect.get_or_raise = tracked_get_or_raise

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
    results = []  # list of (nodeid, lines_list, fixtures_list, dialects_list)
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
            active_test = False
            if hasattr(cls, "setUpClass"):
                try:
                    cls.setUpClass()
                except Exception:
                    pass
            saved_class_fixtures = set(class_fixtures)
            saved_class_dialects = set(class_dialects)

            for meth_name in meths:
                nodeid = f"{file_path}::{cls_name}::{meth_name}"
                current_lines.clear()
                current_fixtures.clear()
                current_dialects.clear()
                inst = cls(meth_name)
                active_test = True
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
                all_fx = set(current_fixtures)
                for fx in saved_class_fixtures:
                    if cls_name == "TestExecutor":
                        if "tpc-h" in fx and "tpch" not in meth_name:
                            continue
                        if "tpc-ds" in fx and "tpcds" not in meth_name:
                            continue
                    all_fx.add(fx)
                all_dl = sorted(current_dialects | saved_class_dialects)
                results.append((nodeid, [f"{r}:{ln}" for r, ln in current_lines], sorted(all_fx), all_dl))

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

    builtins.open = orig_open
    gzip.open = orig_gzip_open
    Dialect.get_or_raise = orig_get_or_raise
    return results


def build_all(repo_root: Path, out_path: Path, workers: int | None = None) -> None:
    items = _discover_unittest_items(repo_root)
    n_workers = workers or min(8, os.cpu_count() or 4)
    buckets_map = [defaultdict(list) for _ in range(n_workers)]
    # Distribute individual test methods round-robin across workers so heavy files parallelize!
    for i, (rel, cls_name, meth_name) in enumerate(items):
        buckets_map[i % n_workers][rel].append((cls_name, meth_name))
    buckets = [sorted(b.items()) for b in buckets_map if b]

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        worker_outputs = pool.map(_trace_worker, [(str(repo_root), b) for b in buckets if b])

    nodeids = []
    nodeid_to_idx = {}
    line_to_idxs = defaultdict(list)
    fixture_to_idxs = defaultdict(list)
    dialect_to_idxs = defaultdict(list)

    for wout in worker_outputs:
        for nodeid, lines, fxs, dls in wout:
            idx = len(nodeids)
            nodeids.append(nodeid)
            nodeid_to_idx[nodeid] = idx
            for loc in lines:
                line_to_idxs[loc].append(idx)
            for fx in fxs:
                fixture_to_idxs[fx].append(idx)
            for dl in dls:
                dialect_to_idxs[dl].append(idx)

    # Static AST + inheritance analysis
    sys.path.insert(0, str(repo_root))
    from sqlglot.dialects.dialect import Dialect, Dialects

    dialect_subclasses = defaultdict(set)
    parser_subclasses = defaultdict(set)
    generator_subclasses = defaultdict(set)
    # Walk all registered dialects in Dialect._classes plus Dialects enum
    seen_dialect_cls = set()
    for d in Dialects:
        inst = Dialect.get_or_raise(d)
        seen_dialect_cls.add(type(inst))
    for cls_obj in Dialect.__subclasses__():
        seen_dialect_cls.add(cls_obj)
    # BFS subclasses
    stack = list(seen_dialect_cls)
    while stack:
        c = stack.pop()
        for sub in c.__subclasses__():
            if sub not in seen_dialect_cls:
                seen_dialect_cls.add(sub)
                stack.append(sub)

    for d_cls in seen_dialect_cls:
        d_name = d_cls.__name__
        try:
            inst = d_cls()
            p_cls = type(inst.parser())
            g_cls = type(inst.generator())
        except Exception:
            p_cls = None
            g_cls = None
        for base in d_cls.__mro__:
            if issubclass(base, Dialect):
                dialect_subclasses[base.__name__].add(d_name)
        if p_cls is not None:
            for base in p_cls.__mro__:
                if base.__name__.endswith("Parser"):
                    parser_subclasses[base.__name__].add(d_name)
        if g_cls is not None:
            for base in g_cls.__mro__:
                if base.__name__.endswith("Generator"):
                    generator_subclasses[base.__name__].add(d_name)

    mod_to_file = {}
    file_data = {}
    attr_readers = defaultdict(list)
    name_readers = defaultdict(list)

    for p in sorted((repo_root / "sqlglot").rglob("*.py")):
        rel = p.relative_to(repo_root).as_posix()
        src = p.read_text(encoding="utf-8")
        info = analyze_python_file(rel, src)
        file_data[rel] = info
        mod_to_file[info["module"]] = rel
        for attr, lns in info["attr_reads"].items():
            for ln in lns:
                attr_readers[attr].append(f"{rel}:{ln}")
        for nm, lns in info["name_reads"].items():
            for ln in lns:
                name_readers[nm].append(f"{rel}:{ln}")

    # Strip attr_reads / name_reads from file_data to keep map compact
    compact_file_data = {}
    for rel, info in file_data.items():
        compact_file_data[rel] = {
            "module": info["module"],
            "imports": info["imports"],
            "line_info": info["line_info"],
            "stmt_spans": info["stmt_spans"],
        }

    test_file_imports = {}
    for p in sorted((repo_root / "tests").rglob("test_*.py")):
        tf = p.relative_to(repo_root).as_posix()
        info = analyze_python_file(tf, p.read_text(encoding="utf-8"))
        direct = {mod_to_file[imp] for imp in info["imports"] if imp in mod_to_file}
        seen = set(direct)
        st = list(direct)
        while st:
            cur = st.pop()
            for imp in file_data[cur]["imports"]:
                target = mod_to_file.get(imp)
                if target and target not in seen:
                    seen.add(target)
                    st.append(target)
        test_file_imports[tf] = sorted(seen)

    payload = {
        "nodeids": nodeids,
        "line_to_idxs": dict(line_to_idxs),
        "fixture_to_idxs": dict(fixture_to_idxs),
        "dialect_to_idxs": dict(dialect_to_idxs),
        "dialect_subclasses": {k: sorted(v) for k, v in dialect_subclasses.items()},
        "parser_subclasses": {k: sorted(v) for k, v in parser_subclasses.items()},
        "generator_subclasses": {k: sorted(v) for k, v in generator_subclasses.items()},
        "file_data": compact_file_data,
        "attr_readers": dict(attr_readers),
        "name_readers": dict(name_readers),
        "test_file_imports": test_file_imports,
    }

    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"Built {out_path} ({out_path.stat().st_size / 1e6:.2f} MB) for {len(nodeids)} tests", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: build_maps.py <sqlglot_repo_dir> <out_maps.json.gz>", file=sys.stderr)
        sys.exit(2)
    build_all(Path(sys.argv[1]), Path(sys.argv[2]))
