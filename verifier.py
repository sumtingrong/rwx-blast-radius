"""Isolated in-process dynamic test verifier with subtest-level acceleration and instant fast-fail."""
from __future__ import annotations

import builtins
import concurrent.futures
import gzip
import importlib.abc
import importlib.machinery
import io
import json
import os
import pickle
import signal
import sys
import time
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


def _stage_a_est_cpu(nid: str, dur_ms_val: int, cnt: int, total: int, done_files: set[str]) -> float:
    fpath = nid.split("::", 1)[0]
    if fpath == "tests/test_executor.py":
        setup = 0.0 if fpath in done_files else 3200.0
        scale = 1.5
    elif fpath == "tests/test_optimizer.py":
        setup = 0.0 if fpath in done_files else 800.0
        scale = 2.5
    else:
        setup = 50.0
        scale = 1.1
    return setup + dur_ms_val * scale * (cnt / total)


def compute_subtest_plan(
    sel: set[int],
    touch_info: dict,
    subtest_map: dict,
    num_to_id: dict[int, str],
    dur_ms: dict[int, int],
    pack_cost_fn,
) -> list[tuple[int, str, str, list[int], int]]:
    """Return eligible (test_num, nodeid, kind, sub_idxs, total) candidates sorted slowest-first."""
    if not subtest_map or not touch_info.get("subtest_eligible"):
        return []
    t_lines = touch_info.get("touched_lines", ())
    t_dicts = touch_info.get("touched_dicts", ())
    t_attrs = touch_info.get("touched_attrs", ())
    if not (t_lines or t_dicts or t_attrs):
        return []

    candidates = []
    for num in sorted(sel, key=lambda n: -dur_ms[n]):
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
        cnt = sub_mask.bit_count()
        if cnt >= max(1, int(0.45 * total)):
            continue
        sub_idxs = [i for i in range(total) if (sub_mask >> i) & 1]
        candidates.append((num, nid, srec["kind"], sub_idxs, total))

    if not candidates:
        return []

    # Check whether at least one candidate is profitable under optimistic cascading removal
    sim_sel = set(sel)
    sim_done: set[str] = set()
    any_profitable = False
    for num, nid, kind, sub_idxs, total in candidates:
        est_cpu = _stage_a_est_cpu(nid, dur_ms[num], len(sub_idxs), total, sim_done)
        delta_w = pack_cost_fn(sim_sel) - pack_cost_fn(sim_sel - {num})
        if delta_w > 0 and est_cpu <= 0.35 * delta_w and est_cpu <= 5500.0:
            any_profitable = True
            sim_sel.discard(num)
            sim_done.add(nid.split("::", 1)[0])
    return candidates if any_profitable else []


def should_zero_fail_prove(
    sel: set[int],
    num_to_id: dict[int, str],
    dur_ms: dict[int, int],
    pack_cost_fn,
    setup_done_files: set[str] | None = None,
) -> bool:
    if not sel:
        return False
    done_files = setup_done_files or set()
    max_single = 0
    t_sum = 0
    for num in sel:
        nid = num_to_id.get(num, "")
        if not nid or is_unverifiable(nid):
            return False
        fpath = nid.split("::", 1)[0]
        # Exclude heavy pandas/duckdb test modules from Stage B unless already initialized in this fork
        if fpath in ("tests/test_executor.py", "tests/test_optimizer.py") and fpath not in done_files:
            return False
        d = dur_ms[num]
        t_sum += d
        if d > max_single:
            max_single = d
    if max_single > 1000:
        return False
    return t_sum <= 2000


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
    sub_spec: tuple[str, set[int]] | None = None,
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
        signal.setitimer(signal.ITIMER_REAL, max(1.5, timeout_s))

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
            kind, allowed = sub_spec
            if kind == "mp_exec":
                orig_mpe = cls._mp_execute

                def _filtered_mpe(self_inst, schema, tables, sqls, tpch):
                    mod.initializer(schema, tables)
                    for idx_q, (q_args, q_sql, _) in enumerate(sqls):
                        if idx_q not in allowed:
                            continue
                        p_ast = pickle.loads(pickle.dumps(mod.parse_one(q_sql)))
                        p_args = pickle.loads(pickle.dumps(q_args))
                        table = mod.mp_execute(p_ast, p_args)
                        table = pickle.loads(pickle.dumps(table))
                        if table is not None:
                            self_inst.subtestHelper(idx_q, table, tpch=tpch)

                cls._mp_execute = _filtered_mpe
                restorers.append(lambda: setattr(cls, "_mp_execute", orig_mpe))
            elif kind == "opt_tpch":
                inst.tpch_sqls = [item for idx_q, item in enumerate(cls.tpch_sqls) if idx_q in allowed]
            elif kind == "gen_mem":
                orig_dialects = mod.Dialects
                mod.Dialects = [d for idx_d, d in enumerate(orig_dialects) if idx_d in allowed]
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
    subtest_plan: list[tuple[int, str, str, list[int], int]],
    all_candidates: list[tuple[int, str, int]],
    allow_zero_fail: bool,
    pack_cost_fn,
    dur_ms: dict[int, int],
    num_to_id: dict[int, str],
) -> tuple[list[int], list[int]]:
    _install_overlays(ref_root, overlay)
    loaded_mods: dict = {}
    loaded_classes: dict = {}
    setup_done: set = set()
    setup_failed: set = set()

    passed: list[int] = []
    failed: list[int] = []
    passed_set: set[int] = set()
    cur_sel: set[int] = {num for num, _, _ in all_candidates}
    any_failed = False

    # Stage A: Subtest-accelerated verification of heavy tests with dynamic delta_w & setup reuse
    for num, nid, kind, sub_idxs, total in subtest_plan:
        if num not in cur_sel:
            continue
        done_files = {m.replace(".", "/") + ".py" for m, _ in setup_done}
        est_cpu_ms = _stage_a_est_cpu(nid, dur_ms[num], len(sub_idxs), total, done_files)
        delta_w = pack_cost_fn(cur_sel) - pack_cost_fn(cur_sel - {num})
        if delta_w <= 0 or est_cpu_ms > 0.35 * delta_w or est_cpu_ms > 5500.0:
            continue
        timeout_s = max(4.0, min(20.0, 4.0 * (est_cpu_ms / 1000.0)))
        status = _run_one_test(
            nid,
            timeout_s,
            loaded_mods,
            loaded_classes,
            setup_done,
            setup_failed,
            sub_spec=(kind, set(sub_idxs)),
        )
        if status == "pass":
            passed.append(num)
            passed_set.add(num)
            cur_sel.discard(num)
        elif status == "fail":
            failed.append(num)
            any_failed = True

    # Stage B: Zero-fail proving on remaining candidates (only if no failure seen yet)
    if allow_zero_fail and not any_failed and cur_sel:
        done_files = {m.replace(".", "/") + ".py" for m, _ in setup_done}
        if should_zero_fail_prove(cur_sel, num_to_id, dur_ms, pack_cost_fn, setup_done_files=done_files):
            rem_ordered = [
                (num, nid, d_ms)
                for num, nid, d_ms in sorted(all_candidates, key=lambda x: x[2])
                if num in cur_sel
            ]
            for num, nid, d_ms in rem_ordered:
                timeout_s = max(2.0, min(15.0, 4.0 * (d_ms / 1000.0)))
                status = _run_one_test(
                    nid,
                    timeout_s,
                    loaded_mods,
                    loaded_classes,
                    setup_done,
                    setup_failed,
                    sub_spec=None,
                )
                if status == "pass":
                    passed.append(num)
                    passed_set.add(num)
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
    subtest_plan: list[tuple[int, str, str, list[int], int]],
    all_candidates: list[tuple[int, str, int]],
    allow_zero_fail: bool,
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
                subtest_plan,
                all_candidates,
                allow_zero_fail,
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
