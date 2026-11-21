"""Isolated in-process dynamic test verifier with lazy TestExecutor, DuckDB cache, subtest acceleration, and instant fast-fail."""
from __future__ import annotations

import ast
import builtins
import concurrent.futures
import csv
import gzip
import importlib.abc
import importlib.machinery
import io
import json
import os
import pickle
import signal
import sys
import unittest
from concurrent.futures import Future
from pathlib import Path

try:
    import resource
except ImportError:
    resource = None

_UNVERIFIABLE_IDS = frozenset({
    "tests/dialects/test_dialect.py::TestDialect::test_lazy_load",
})
_UNVERIFIABLE_FILES = frozenset({
    "tests/test_sqlglotc.py",
})

_TPCDS_TID = "tests/test_executor.py::TestExecutor::test_execute_tpcds"
_OPT_TPCH_TID = "tests/test_executor.py::TestExecutor::test_optimized_tpch"

# Exact set of files loaded by `import sqlglot`. Any commit that touches ONLY
# files outside this set can safely run in a warm fork after `import sqlglot`.
_CORE_IMPORT_RELS = frozenset({
    "sqlglot/__init__.py",
    "sqlglot/_typing.py",
    "sqlglot/dialects/__init__.py",
    "sqlglot/dialects/dialect.py",
    "sqlglot/diff.py",
    "sqlglot/errors.py",
    "sqlglot/expressions/__init__.py",
    "sqlglot/expressions/aggregate.py",
    "sqlglot/expressions/array.py",
    "sqlglot/expressions/builders.py",
    "sqlglot/expressions/constraints.py",
    "sqlglot/expressions/core.py",
    "sqlglot/expressions/datatypes.py",
    "sqlglot/expressions/ddl.py",
    "sqlglot/expressions/dml.py",
    "sqlglot/expressions/functions.py",
    "sqlglot/expressions/json.py",
    "sqlglot/expressions/math.py",
    "sqlglot/expressions/properties.py",
    "sqlglot/expressions/query.py",
    "sqlglot/expressions/string.py",
    "sqlglot/expressions/temporal.py",
    "sqlglot/generator.py",
    "sqlglot/helper.py",
    "sqlglot/jsonpath.py",
    "sqlglot/optimizer/__init__.py",
    "sqlglot/optimizer/scope.py",
    "sqlglot/parser.py",
    "sqlglot/parsers/__init__.py",
    "sqlglot/parsers/base.py",
    "sqlglot/schema.py",
    "sqlglot/time.py",
    "sqlglot/tokenizer_core.py",
    "sqlglot/tokens.py",
    "sqlglot/trie.py",
    "sqlglot/typing/__init__.py",
})

# Fallback CPU ratios when test_cpu.pkl is absent (e.g. legacy maps)
_FALLBACK_TEST_CPU_MS = {
    "tests/test_parser.py::TestParser::test_max_nodes": 35000,
    "tests/test_executor.py::TestExecutor::test_execute_tpcds": 18000,
    "tests/test_generator.py::TestGenerator::test_generator_memory": 1600,
    "tests/test_executor.py::TestExecutor::test_execute_tpch": 70,
    "tests/test_executor.py::TestExecutor::test_optimized_tpch": 50,
    "tests/test_executor.py::TestExecutor::test_agg_order": 5,
    "tests/test_optimizer.py::TestOptimizer::test_canonicalize": 100,
    "tests/test_schema.py::TestSchema::test_cache_does_not_retain_queries": 250,
    "tests/test_optimizer.py::TestOptimizer::test_values_annotation": 20,
    "tests/dialects/test_dialect.py::TestDialect::test_time": 40,
}


def is_unverifiable(tid: str) -> bool:
    if tid in _UNVERIFIABLE_IDS:
        return True
    fpath = tid.split("::", 1)[0]
    return fpath in _UNVERIFIABLE_FILES


def needs_clean_fork(overlay: dict[str, str | None]) -> bool:
    for rel in overlay:
        if rel in _CORE_IMPORT_RELS:
            return True
    return False


def can_lazy_executor(overlay: dict[str, str | None]) -> bool:
    for rel in overlay:
        if rel in ("sqlglot/executor/table.py", "tests/test_executor.py", "tests/helpers.py"):
            return False
        if rel.startswith("tests/fixtures/optimizer/tpc-h/") or rel.startswith("tests/fixtures/optimizer/tpc-ds/"):
            return False
    return True


def build_patch_overlay(
    patch_text: str,
    ref_root: Path,
    parse_patch_fn,
    apply_hunks_fn,
) -> tuple[dict[str, str | None], bool]:
    if "Binary files " in patch_text or "GIT binary patch" in patch_text:
        return {}, False
    overlay: dict[str, str | None] = {}
    for fd in parse_patch_fn(patch_text):
        old_p = fd["old_path"]
        new_p = fd["new_path"]
        if old_p and new_p and old_p != new_p:
            return {}, False
        rel = new_p or old_p
        if not rel:
            continue
        if fd["deleted_file"]:
            overlay[rel] = None
        elif fd["new_file"]:
            out_lines: list[str] = []
            for h in fd["hunks"]:
                for hl in h["lines"]:
                    if hl.startswith("+"):
                        out_lines.append(hl[1:])
                    elif hl.startswith("\\") and out_lines and out_lines[-1].endswith("\n"):
                        out_lines[-1] = out_lines[-1][:-1]
            overlay[rel] = "".join(out_lines)
        else:
            p = ref_root / rel
            if not p.exists():
                return {}, False
            try:
                orig_txt = p.read_text("utf-8")
            except Exception:
                return {}, False
            patched = apply_hunks_fn(orig_txt, fd["hunks"])
            if patched is None:
                return {}, False
            overlay[rel] = patched
    return overlay, True


def overlay_mod_names(overlay: dict[str, str | None]) -> set[str]:
    mods: set[str] = set()
    for rel in overlay:
        if rel.endswith(".py"):
            mname = rel[:-12].replace("/", ".") if rel.endswith("/__init__.py") else rel[:-3].replace("/", ".")
            mods.add(mname)
    return mods


def compute_subtest_elig(
    sel: set[int],
    touch_info: dict,
    subtest_map: dict,
    cpu_info: dict,
    num_to_id: dict[int, str],
) -> dict[int, tuple[str, list[int], int]]:
    """Return per-test eligible (kind, sub_idxs_fastest_first, total) for tests in sel."""
    if not subtest_map or touch_info.get("subtest_global_block"):
        return {}
    if touch_info.get("subtest_eligible") is False:
        return {}
    t_lines = touch_info.get("touched_lines", ())
    t_dicts = touch_info.get("touched_dicts", ())
    t_attrs = touch_info.get("touched_attrs", ())
    if not (t_lines or t_dicts or t_attrs):
        return {}
    blocked = touch_info.get("subtest_blocked_tests", ())
    sub_times_all = cpu_info.get("sub_times_ms", {})

    elig: dict[int, tuple[str, list[int], int]] = {}
    for num in sel:
        if num in blocked:
            continue
        nid = num_to_id.get(num)
        if not nid or nid not in subtest_map or is_unverifiable(nid):
            continue
        srec = subtest_map[nid]
        total = srec.get("total", 0)
        if total <= 1:
            continue
        s_lines = srec.get("lines", {})
        s_dicts = srec.get("dicts", {})
        s_attrs = srec.get("attrs", {})
        union_mask = 0
        for rel, ln in t_lines:
            rmap = s_lines.get(rel)
            if rmap:
                union_mask |= rmap.get(ln, 0)
        for dk in t_dicts:
            union_mask |= s_dicts.get(dk, 0)
        for ak in t_attrs:
            union_mask |= s_attrs.get(ak, 0)
        # PRE_LOOP bit (bit 0) means hit outside subtest loop or in setUpClass -> cannot filter safely
        if union_mask & 1:
            continue
        sub_mask = union_mask >> 1
        if not sub_mask:
            continue
        sub_idxs = [i for i in range(total) if (sub_mask >> i) & 1]
        if not sub_idxs:
            continue
        stimes = sub_times_all.get(nid)
        if stimes and len(stimes) >= total:
            sub_idxs.sort(key=lambda i: (stimes[i], i))
        elig[num] = (srec["kind"], sub_idxs, total)
    return elig


def _file_setup_ms(fpath: str, done_files: set[str], cpu_info: dict, can_lazy: bool) -> float:
    if fpath in done_files:
        return 0.0
    heavy_pre = bool(cpu_info.get("heavy_preimported"))
    if fpath == "tests/test_executor.py":
        imp = 0.0 if heavy_pre else 550.0
        return imp + (5.0 if can_lazy else 2000.0)
    if fpath == "tests/test_optimizer.py":
        return 60.0 if heavy_pre else 610.0
    return 15.0


def _est_body_cpu_ms(
    num: int,
    nid: str,
    dur_ms_val: int,
    sub_entry: tuple[str, list[int], int] | None,
    cpu_info: dict,
    can_lazy: bool,
    loaded_tpcds_tables: set[str],
    has_duckdb_cache: bool,
) -> float:
    test_cpu_ms = cpu_info.get("test_cpu_ms", {})
    sub_times_all = cpu_info.get("sub_times_ms", {})
    tpcds_tbls = cpu_info.get("tpcds_tables_by_idx", {})
    tpcds_csv_ms = cpu_info.get("tpcds_csv_ms", {})

    if sub_entry is not None:
        _, sub_idxs, total = sub_entry
        stimes = sub_times_all.get(nid)
        if stimes and len(stimes) >= total:
            body = float(sum(stimes[i] for i in sub_idxs))
            if nid == _TPCDS_TID and not has_duckdb_cache:
                body *= 2.5
        else:
            full = float(test_cpu_ms.get(nid, _FALLBACK_TEST_CPU_MS.get(nid, dur_ms_val * 0.9)))
            body = full * (len(sub_idxs) / max(1, total))
        if nid == _TPCDS_TID and can_lazy:
            needed: set[str] = set()
            for i in sub_idxs:
                needed.update(tpcds_tbls.get(i, ()))
            for tbl in needed - loaded_tpcds_tables:
                body += float(tpcds_csv_ms.get(tbl, 60))
        return max(1.0, body)

    if nid == _OPT_TPCH_TID and not (can_lazy and has_duckdb_cache):
        return 5500.0
    if nid == _TPCDS_TID:
        base = float(test_cpu_ms.get(nid, _FALLBACK_TEST_CPU_MS.get(nid, 18000)))
        if not has_duckdb_cache:
            base *= 2.5
        if can_lazy:
            all_t: set[str] = set()
            for ts in tpcds_tbls.values():
                all_t.update(ts)
            for tbl in all_t - loaded_tpcds_tables:
                base += float(tpcds_csv_ms.get(tbl, 60))
        return base
    return max(1.0, float(test_cpu_ms.get(nid, _FALLBACK_TEST_CPU_MS.get(nid, dur_ms_val * 0.9))))


def _can_zero_fail_prove(
    cur_sel: set[int],
    subtest_elig: dict[int, tuple[str, list[int], int]],
    cpu_info: dict,
    num_to_id: dict[int, str],
    dur_ms: dict[int, int],
    pack_cost_fn,
    done_files: set[str],
    loaded_tpcds_tables: set[str],
    can_lazy: bool,
    has_duckdb_cache: bool,
) -> bool:
    if not cur_sel:
        return False
    cur_cost = pack_cost_fn(cur_sel)
    if cur_cost <= 0:
        return False
    sim_done = set(done_files)
    sim_tbls = set(loaded_tpcds_tables)
    tot_cpu = 0.0
    max_single = 0.0
    tpcds_tbls = cpu_info.get("tpcds_tables_by_idx", {})
    for num in cur_sel:
        nid = num_to_id.get(num, "")
        if not nid or is_unverifiable(nid):
            return False
        fpath = nid.split("::", 1)[0]
        setup = _file_setup_ms(fpath, sim_done, cpu_info, can_lazy)
        sim_done.add(fpath)
        sub_e = subtest_elig.get(num)
        body = _est_body_cpu_ms(num, nid, dur_ms[num], sub_e, cpu_info, can_lazy, sim_tbls, has_duckdb_cache)
        if nid == _TPCDS_TID and can_lazy:
            if sub_e is not None:
                for i in sub_e[1]:
                    sim_tbls.update(tpcds_tbls.get(i, ()))
            else:
                for ts in tpcds_tbls.values():
                    sim_tbls.update(ts)
        tot_cpu += setup + body
        if body > max_single:
            max_single = body
        if max_single > 1800.0 or tot_cpu > 4200.0:
            return False
    return tot_cpu <= min(4200.0, 0.72 * cur_cost)


def plan_commit_verification(
    sel: set[int],
    touch_info: dict,
    subtest_map: dict,
    cpu_info: dict,
    num_to_id: dict[int, str],
    dur_ms: dict[int, int],
    pack_cost_fn,
    can_lazy: bool,
    has_duckdb_cache: bool,
) -> tuple[dict[int, tuple[str, list[int], int]], bool, int]:
    """Decide whether to fork a verifier for this commit; returns (subtest_elig, has_work, priority)."""
    if not sel:
        return {}, False, 0
    subtest_elig = compute_subtest_elig(sel, touch_info, subtest_map, cpu_info, num_to_id)

    # Optimistically simulate Phase 1 to see if any high-ROI pruning candidate exists
    sim_sel = set(sel)
    sim_done: set[str] = set()
    sim_tbls: set[str] = set()
    sim_tested: set[int] = set()
    tpcds_tbls = cpu_info.get("tpcds_tables_by_idx", {})
    phase1_gain = 0

    for _ in range(16):
        best_choice = None
        best_roi = 0.0
        arb_group: set[int] | None = None
        for num in sim_sel:
            if num in sim_tested:
                continue
            nid = num_to_id.get(num, "")
            if not nid or is_unverifiable(nid):
                continue
            fpath = nid.split("::", 1)[0]
            sub_e = subtest_elig.get(num)
            est_cpu = _file_setup_ms(fpath, sim_done, cpu_info, can_lazy) + _est_body_cpu_ms(
                num, nid, dur_ms[num], sub_e, cpu_info, can_lazy, sim_tbls, has_duckdb_cache
            )
            if est_cpu > 9500.0:
                continue
            delta_w = pack_cost_fn(sim_sel) - pack_cost_fn(sim_sel - {num})
            eff_delta = delta_w
            if eff_delta <= 0 and est_cpu <= 400.0 and dur_ms[num] >= 500 and dur_ms[num] >= 3.5 * est_cpu:
                if arb_group is None:
                    arb_items = []
                    for x in sim_sel:
                        if x in sim_tested or dur_ms[x] < 500:
                            continue
                        xnid = num_to_id.get(x)
                        if not xnid or is_unverifiable(xnid):
                            continue
                        xcpu = _file_setup_ms(xnid.split("::", 1)[0], sim_done, cpu_info, can_lazy) + _est_body_cpu_ms(
                            x, xnid, dur_ms[x], subtest_elig.get(x), cpu_info, can_lazy, sim_tbls, has_duckdb_cache
                        )
                        if xcpu <= 400.0 and dur_ms[x] >= 3.5 * xcpu:
                            arb_items.append((x, xcpu))
                    arb_set = {x for x, _ in arb_items}
                    g_delta = (pack_cost_fn(sim_sel) - pack_cost_fn(sim_sel - arb_set)) if arb_set else 0
                    g_cpu = sum(c for _, c in arb_items)
                    arb_group = arb_set if (g_delta > 0 and g_cpu <= 0.65 * g_delta) else set()
                if num in arb_group:
                    eff_delta = dur_ms[num]
            if eff_delta > 0 and est_cpu <= 0.50 * eff_delta:
                roi = eff_delta / max(10.0, est_cpu)
                if roi > best_roi:
                    best_roi = roi
                    best_choice = (num, nid, fpath, sub_e, delta_w)
        if best_choice is None:
            break
        num, nid, fpath, sub_e, delta_w = best_choice
        sim_tested.add(num)
        sim_sel.discard(num)
        sim_done.add(fpath)
        if nid == _TPCDS_TID and can_lazy:
            if sub_e is not None:
                for i in sub_e[1]:
                    sim_tbls.update(tpcds_tbls.get(i, ()))
            else:
                for ts in tpcds_tbls.values():
                    sim_tbls.update(ts)
        phase1_gain += max(0, delta_w)

    can_zf = _can_zero_fail_prove(
        sim_sel, subtest_elig, cpu_info, num_to_id, dur_ms, pack_cost_fn, sim_done, sim_tbls, can_lazy, has_duckdb_cache
    )
    has_work = (phase1_gain > 0) or bool(sim_tested) or can_zf
    priority = phase1_gain + (pack_cost_fn(sim_sel) if can_zf else 0)
    return subtest_elig, has_work, priority


class _SyncPool:
    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
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
        except BaseException as e:
            fut.set_exception(e)
        return fut

    def map(self, fn, iterable, *a, **kw):
        return list(map(fn, iterable))

    def shutdown(self, wait=True, cancel_futures=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FastFail(BaseException):
    pass


class _TestTimeout(BaseException):
    pass


class _FastFailResult(unittest.TestResult):
    def addError(self, test, err):
        super().addError(test, err)
        raise _FastFail()

    def addFailure(self, test, err):
        super().addFailure(test, err)
        raise _FastFail()

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            raise _FastFail()

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        raise _FastFail()


class _LazyTableDict(dict):
    """Loads individual TPC-H / TPC-DS .csv.gz tables strictly on first access."""

    def __init__(self, mod, directory: str, schema: dict):
        super().__init__()
        self._mod = mod
        self._directory = directory
        self._schema = schema

    def __missing__(self, table: str):
        columns = self._schema[table]
        file_name = f"{self._directory}{table}.csv.gz"
        reader = csv.reader(self._mod.open_file(file_name), delimiter="|")
        rows = []
        ctypes = []
        next(reader)
        for row in reader:
            if not ctypes:
                for v in row:
                    try:
                        ctypes.append(type(ast.literal_eval(v)))
                    except (ValueError, SyntaxError):
                        ctypes.append(str)
            rows.append(
                tuple(None if (t is not str and v == "") else t(v) for t, v in zip(ctypes, row))
            )
        tbl = self._mod.Table(columns=columns, rows=rows)
        self[table] = tbl
        return tbl


def _patch_lazy_executor(mod, cls, duckdb_cache: dict | None) -> None:
    """Install lazy CSV/DuckDB setup + sound pinned-commit DuckDB cache on TestExecutor."""
    opt_tpch_cache = (duckdb_cache or {}).get("opt_tpch") or []
    tpcds_df_by_sql = (duckdb_cache or {}).get("tpcds_df_by_sql") or {}

    def _ensure_conn(target_cls, tpch: bool):
        attr = "_lazy_tpch_conn" if tpch else "_lazy_tpcds_conn"
        conn = getattr(target_cls, attr, None)
        if conn is None:
            conn = mod.duckdb.connect()
            directory = mod.DIR_TPCH if tpch else mod.DIR_TPCDS
            schema = mod.TPCH_SCHEMA if tpch else mod.TPCDS_SCHEMA
            for table, columns in schema.items():
                file_name = f"{directory}{table}.csv.gz"
                conn.execute(
                    f"CREATE VIEW {table} AS SELECT * FROM READ_CSV('{file_name}', delim='|', header=True, columns={columns})"
                )
            setattr(target_cls, attr, conn)
        return conn

    @classmethod
    def _lazy_setup_class(target_cls):
        target_cls._lazy_tpch_conn = None
        target_cls._lazy_tpcds_conn = None
        target_cls.tpch_tables = _LazyTableDict(mod, mod.DIR_TPCH, mod.TPCH_SCHEMA)
        target_cls.tpcds_tables = _LazyTableDict(mod, mod.DIR_TPCDS, mod.TPCDS_SCHEMA)
        target_cls.cache = {}
        target_cls.tpch_sqls = list(mod.load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql"))
        target_cls.tpcds_sqls = list(mod.load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))

    @classmethod
    def _lazy_teardown_class(target_cls):
        for attr in ("_lazy_tpch_conn", "_lazy_tpcds_conn"):
            c = getattr(target_cls, attr, None)
            if c is not None:
                try:
                    c.close()
                except Exception:
                    pass
                setattr(target_cls, attr, None)

    def _lazy_cached_execute(self, sql, tpch=True):
        if sql not in self.cache:
            dsql = mod.transpile(sql, write="duckdb")[0]
            if not tpch and sql in tpcds_df_by_sql:
                base_dsql, base_df = tpcds_df_by_sql[sql]
                if dsql == base_dsql:
                    self.cache[sql] = base_df.copy()
                    return self.cache[sql]
            conn = _ensure_conn(type(self), tpch=tpch)
            self.cache[sql] = conn.execute(dsql).fetchdf()
        return self.cache[sql]

    def _cached_test_optimized_tpch(self):
        allowed = getattr(self, "_opt_tpch_allowed", None)
        indices = allowed if allowed is not None else range(len(self.tpch_sqls))
        for idx in indices:
            _, sql, optimized = self.tpch_sqls[idx]
            with self.subTest(f"{idx + 1}, {sql}"):
                s1 = mod.transpile(sql, write="duckdb")[0]
                s2 = mod.transpile(optimized, write="duckdb")[0]
                if idx < len(opt_tpch_cache) and (s1, s2) == opt_tpch_cache[idx]:
                    continue
                a = self.cached_execute(sql, tpch=True)
                conn = _ensure_conn(type(self), tpch=True)
                b = conn.execute(s2).fetchdf()
                self.rename_anonymous(b, a)
                mod.assert_frame_equal(a, b)

    cls.setUpClass = _lazy_setup_class
    cls.tearDownClass = _lazy_teardown_class
    cls.tpch_conn = property(lambda self: _ensure_conn(type(self), tpch=True))
    cls.tpcds_conn = property(lambda self: _ensure_conn(type(self), tpch=False))
    cls.cached_execute = _lazy_cached_execute
    if opt_tpch_cache:
        cls.test_optimized_tpch = _cached_test_optimized_tpch


def _install_overlays(ref_root: Path, overlay: dict[str, str | None]) -> None:
    os.chdir(ref_root)
    ref_str = str(ref_root)
    if ref_str not in sys.path:
        sys.path.insert(0, ref_str)

    concurrent.futures.ProcessPoolExecutor = _SyncPool

    mod_overlay: dict[str, tuple[str, str | None, bool]] = {}
    for rel, content in overlay.items():
        if rel.endswith(".py"):
            is_pkg = rel.endswith("/__init__.py")
            mname = rel[:-12].replace("/", ".") if is_pkg else rel[:-3].replace("/", ".")
            mod_overlay[mname] = (rel, content, is_pkg)

    if mod_overlay:
        class _OverlayLoader(importlib.abc.SourceLoader):
            def __init__(self, fullname: str, path: str, content: str):
                self._fullname = fullname
                self._path = path
                self._content = content.encode("utf-8")

            def get_filename(self, fullname: str) -> str:
                return self._path

            def get_data(self, path: str) -> bytes:
                if path == self._path:
                    return self._content
                return Path(path).read_bytes()

        class _OverlayFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                entry = mod_overlay.get(fullname)
                if entry is not None:
                    rel, content, is_pkg = entry
                    if content is None:
                        raise ModuleNotFoundError(f"Deleted module {fullname}", name=fullname)
                    fpath = str((ref_root / rel).resolve())
                    loader = _OverlayLoader(fullname, fpath, content)
                    spec = importlib.machinery.ModuleSpec(fullname, loader, origin=fpath, is_package=is_pkg)
                    if is_pkg:
                        spec.submodule_search_locations = [str((ref_root / rel).parent.resolve())]
                    return spec
                return None

        sys.meta_path.insert(0, _OverlayFinder())

    if overlay:
        orig_open = builtins.open
        orig_gzip_open = gzip.open

        def _match_overlay(file):
            try:
                p = os.fspath(file)
            except TypeError:
                return None
            if not isinstance(p, str):
                return None
            norm = p.replace("\\", "/")
            for rel, content in overlay.items():
                if norm == rel or norm.endswith("/" + rel):
                    return (rel, content)
            return None

        def _patched_open(file, mode="r", *args, **kwargs):
            m = _match_overlay(file)
            if m is not None and "w" not in mode and "a" not in mode and "x" not in mode:
                rel, content = m
                if content is None:
                    raise FileNotFoundError(rel)
                if "b" in mode:
                    return io.BytesIO(content.encode("utf-8"))
                return io.StringIO(content)
            return orig_open(file, mode, *args, **kwargs)

        def _patched_gzip_open(filename, mode="rb", *args, **kwargs):
            m = _match_overlay(filename)
            if m is not None and "w" not in mode and "a" not in mode and "x" not in mode:
                rel, content = m
                if content is None:
                    raise FileNotFoundError(rel)
                return io.StringIO(content) if "t" in mode else io.BytesIO(content.encode("utf-8"))
            return orig_gzip_open(filename, mode, *args, **kwargs)

        builtins.open = _patched_open
        io.open = _patched_open
        gzip.open = _patched_gzip_open


def _run_one_test(
    tid: str,
    timeout_s: float,
    loaded_mods: dict,
    loaded_classes: dict,
    setup_done: set,
    setup_failed: set,
    can_lazy: bool,
    duckdb_cache: dict | None,
    sub_spec: tuple[str, list[int]] | None = None,
) -> str:
    """Run one unittest method; return 'pass', 'fail', or 'skip'."""
    fpath, cls_name, meth_name = tid.split("::")
    mod_name = fpath[:-3].replace("/", ".")
    cls_key = (mod_name, cls_name)
    if cls_key in setup_failed:
        return "fail"

    has_alarm = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")

    def _on_alarm(signum, frame):
        raise _TestTimeout()

    prev_handler = None
    if has_alarm:
        prev_handler = signal.signal(signal.SIGALRM, _on_alarm)
        signal.setitimer(signal.ITIMER_REAL, max(2.0, timeout_s))

    restorers = []
    try:
        if cls_key not in loaded_classes:
            try:
                mod = loaded_mods.get(mod_name)
                if mod is None:
                    mod = __import__(mod_name, fromlist=[cls_name])
                    if hasattr(mod, "ProcessPoolExecutor"):
                        mod.ProcessPoolExecutor = _SyncPool
                    loaded_mods[mod_name] = mod
                cls = getattr(mod, cls_name)
                if can_lazy and mod_name == "tests.test_executor" and cls_name == "TestExecutor":
                    _patch_lazy_executor(mod, cls, duckdb_cache)
                loaded_classes[cls_key] = cls
            except BaseException:
                setup_failed.add(cls_key)
                return "fail"
        else:
            mod = loaded_mods[mod_name]
            cls = loaded_classes[cls_key]

        if cls_key not in setup_done:
            try:
                cls.setUpClass()
                setup_done.add(cls_key)
            except BaseException:
                setup_failed.add(cls_key)
                return "fail"

        inst = cls(meth_name)
        if sub_spec is not None:
            kind, sub_order = sub_spec
            allowed = set(sub_order)
            if kind == "mp_exec":
                orig_mpe = cls._mp_execute

                def _filtered_mpe(self_inst, schema, tables, sqls, tpch):
                    mod.initializer(schema, tables)
                    for idx_q in sub_order:
                        q_args, q_sql, _ = sqls[idx_q]
                        p_ast = pickle.loads(pickle.dumps(mod.parse_one(q_sql)))
                        p_args = pickle.loads(pickle.dumps(q_args))
                        table = mod.mp_execute(p_ast, p_args)
                        table = pickle.loads(pickle.dumps(table))
                        if table is not None:
                            self_inst.subtestHelper(idx_q, table, tpch=tpch)

                cls._mp_execute = _filtered_mpe
                restorers.append(lambda: setattr(cls, "_mp_execute", orig_mpe))
            elif kind == "opt_tpch":
                if can_lazy and duckdb_cache and duckdb_cache.get("opt_tpch"):
                    inst._opt_tpch_allowed = sub_order
                else:
                    inst.tpch_sqls = [cls.tpch_sqls[i] for i in sub_order]
            elif kind == "gen_mem":
                orig_dialects = mod.Dialects
                d_list = list(orig_dialects)
                mod.Dialects = [d_list[i] for i in sub_order]
                restorers.append(lambda: setattr(mod, "Dialects", orig_dialects))
            elif kind in ("check_file", "fixture_pairs"):
                orig_lfp = mod.load_sql_fixture_pairs
                lfp_counter = [0]

                def _filtered_lfp(*a, **kw):
                    for item in orig_lfp(*a, **kw):
                        idx_p = lfp_counter[0]
                        lfp_counter[0] += 1
                        if idx_p in allowed:
                            yield item

                mod.load_sql_fixture_pairs = _filtered_lfp
                restorers.append(lambda: setattr(mod, "load_sql_fixture_pairs", orig_lfp))

        res = _FastFailResult()
        try:
            inst.run(res)
        except (_FastFail, _TestTimeout):
            return "fail"
        except BaseException:
            return "fail"
        if not res.wasSuccessful():
            return "fail"
        if res.skipped:
            return "skip"
        return "pass"
    except (_FastFail, _TestTimeout):
        return "fail"
    except BaseException:
        return "fail"
    finally:
        if has_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            if prev_handler is not None:
                signal.signal(signal.SIGALRM, prev_handler)
        for rfn in restorers:
            try:
                rfn()
            except BaseException:
                pass


def _execute_commit_verification(
    ref_root: Path,
    overlay: dict[str, str | None],
    subtest_elig: dict[int, tuple[str, list[int], int]],
    all_candidates: list[tuple[int, str, int]],
    cpu_info: dict,
    duckdb_cache: dict | None,
    pack_cost_fn,
    dur_ms: dict[int, int],
    num_to_id: dict[int, str],
) -> tuple[list[int], list[int]]:
    _install_overlays(ref_root, overlay)
    can_lazy = can_lazy_executor(overlay)
    has_duckdb_cache = bool(duckdb_cache and duckdb_cache.get("tpcds_df_by_sql"))

    loaded_mods: dict = {}
    loaded_classes: dict = {}
    setup_done: set = set()
    setup_failed: set = set()
    loaded_tpcds_tables: set[str] = set()
    tpcds_tbls = cpu_info.get("tpcds_tables_by_idx", {})

    passed: list[int] = []
    failed: list[int] = []
    tested: set[int] = set()
    cur_sel: set[int] = {num for num, _, _ in all_candidates}
    any_failed = False

    def _done_files() -> set[str]:
        return {m.replace(".", "/") + ".py" for m, _ in setup_done}

    # Phase 1: Dynamic highest-ROI pruning (arbitrage tests + subtest-filtered heavy tests + high-ROI full tests)
    for _ in range(32):
        done_files = _done_files()
        best_choice = None
        best_roi = 0.0
        arb_group: set[int] | None = None

        for num in cur_sel:
            if num in tested:
                continue
            nid = num_to_id.get(num, "")
            if not nid or is_unverifiable(nid):
                continue
            fpath = nid.split("::", 1)[0]
            sub_e = subtest_elig.get(num)
            est_cpu = _file_setup_ms(fpath, done_files, cpu_info, can_lazy) + _est_body_cpu_ms(
                num, nid, dur_ms[num], sub_e, cpu_info, can_lazy, loaded_tpcds_tables, has_duckdb_cache
            )
            if est_cpu > 9500.0:
                continue
            delta_w = pack_cost_fn(cur_sel) - pack_cost_fn(cur_sel - {num})
            eff_delta = delta_w
            if eff_delta <= 0 and est_cpu <= 400.0 and dur_ms[num] >= 500 and dur_ms[num] >= 3.5 * est_cpu:
                if arb_group is None:
                    arb_items = []
                    for x in cur_sel:
                        if x in tested or dur_ms[x] < 500:
                            continue
                        xnid = num_to_id.get(x)
                        if not xnid or is_unverifiable(xnid):
                            continue
                        xcpu = _file_setup_ms(xnid.split("::", 1)[0], done_files, cpu_info, can_lazy) + _est_body_cpu_ms(
                            x, xnid, dur_ms[x], subtest_elig.get(x), cpu_info, can_lazy, loaded_tpcds_tables, has_duckdb_cache
                        )
                        if xcpu <= 400.0 and dur_ms[x] >= 3.5 * xcpu:
                            arb_items.append((x, xcpu))
                    arb_set = {x for x, _ in arb_items}
                    g_delta = (pack_cost_fn(cur_sel) - pack_cost_fn(cur_sel - arb_set)) if arb_set else 0
                    g_cpu = sum(c for _, c in arb_items)
                    arb_group = arb_set if (g_delta > 0 and g_cpu <= 0.65 * g_delta) else set()
                if num in arb_group:
                    eff_delta = dur_ms[num]
            if eff_delta > 0 and est_cpu <= 0.50 * eff_delta:
                roi = eff_delta / max(10.0, est_cpu)
                if roi > best_roi:
                    best_roi = roi
                    best_choice = (num, nid, sub_e, est_cpu)

        if best_choice is None:
            break

        num, nid, sub_e, est_cpu = best_choice
        tested.add(num)
        spec = (sub_e[0], sub_e[1]) if sub_e is not None else None
        timeout_s = max(4.0, min(25.0, 4.0 * (est_cpu / 1000.0)))
        status = _run_one_test(
            nid,
            timeout_s,
            loaded_mods,
            loaded_classes,
            setup_done,
            setup_failed,
            can_lazy,
            duckdb_cache,
            sub_spec=spec,
        )
        if nid == _TPCDS_TID and can_lazy:
            if sub_e is not None:
                for i in sub_e[1]:
                    loaded_tpcds_tables.update(tpcds_tbls.get(i, ()))
            else:
                for ts in tpcds_tbls.values():
                    loaded_tpcds_tables.update(ts)
        if status == "pass":
            passed.append(num)
            cur_sel.discard(num)
        elif status == "fail":
            failed.append(num)
            any_failed = True

    # Phase 2: Expanded Zero-Fail Proving (only if no failure seen yet)
    if not any_failed and cur_sel:
        done_files = _done_files()
        if _can_zero_fail_prove(
            cur_sel,
            subtest_elig,
            cpu_info,
            num_to_id,
            dur_ms,
            pack_cost_fn,
            done_files,
            loaded_tpcds_tables,
            can_lazy,
            has_duckdb_cache,
        ):
            rem_items = []
            for num in cur_sel:
                nid = num_to_id[num]
                sub_e = subtest_elig.get(num)
                b_cpu = _est_body_cpu_ms(
                    num, nid, dur_ms[num], sub_e, cpu_info, can_lazy, loaded_tpcds_tables, has_duckdb_cache
                )
                rem_items.append((b_cpu, num, nid, sub_e))
            rem_items.sort(key=lambda x: (x[0], x[1]))

            for b_cpu, num, nid, sub_e in rem_items:
                spec = (sub_e[0], sub_e[1]) if sub_e is not None else None
                timeout_s = max(2.5, min(15.0, 4.0 * (b_cpu / 1000.0) + 2.0))
                status = _run_one_test(
                    nid,
                    timeout_s,
                    loaded_mods,
                    loaded_classes,
                    setup_done,
                    setup_failed,
                    can_lazy,
                    duckdb_cache,
                    sub_spec=spec,
                )
                if status == "pass":
                    passed.append(num)
                    cur_sel.discard(num)
                elif status == "fail":
                    failed.append(num)
                    any_failed = True
                    break
                else:
                    # Skipped -> cannot prove 0-fail, stop conservatively
                    break

    return passed, failed


def verify_commit_fork(
    ref_root: Path,
    overlay: dict[str, str | None],
    subtest_elig: dict[int, tuple[str, list[int], int]],
    all_candidates: list[tuple[int, str, int]],
    cpu_info: dict,
    duckdb_cache: dict | None,
    pack_cost_fn,
    dur_ms: dict[int, int],
    num_to_id: dict[int, str],
) -> tuple[set[int], set[int]]:
    """Run dynamic verification inside an isolated os.fork() child (Linux)."""
    if not hasattr(os, "fork"):
        return set(), set()
    # Defensive guard: never verify with a stale already-imported module in sys.modules
    if overlay_mod_names(overlay) & sys.modules.keys():
        return set(), set()
    r_fd, w_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r_fd)
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            os.close(devnull)
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")
            passed, failed = _execute_commit_verification(
                ref_root,
                overlay,
                subtest_elig,
                all_candidates,
                cpu_info,
                duckdb_cache,
                pack_cost_fn,
                dur_ms,
                num_to_id,
            )
            data = json.dumps({"p": passed, "f": failed}).encode("utf-8")
            view = memoryview(data)
            written = 0
            while written < len(view):
                written += os.write(w_fd, view[written:])
        except BaseException:
            pass
        finally:
            try:
                os.close(w_fd)
            except OSError:
                pass
            os._exit(0)
    os.close(w_fd)
    chunks = []
    try:
        while True:
            chunk = os.read(r_fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(r_fd)
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
    if not chunks:
        return set(), set()
    try:
        res = json.loads(b"".join(chunks).decode("utf-8"))
        return set(res.get("p", ())), set(res.get("f", ()))
    except Exception:
        return set(), set()


def get_children_cpu_seconds() -> float:
    if resource is None:
        return 0.0
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    return ru.ru_utime + ru.ru_stime
