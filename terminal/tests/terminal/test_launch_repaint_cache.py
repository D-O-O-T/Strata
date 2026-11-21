"""Every repaint of `strata launch` asked the same questions again, so each screen waited on the disk and on the GPU tools.

The screens ask who is installed on every paint (`Catalog.installed_backend`, `Ctx.engine_exe`, the Pictures default), and
`status()` answers it. Before this change every call read every engine's manifest, hashed the entrypoint again, asked
`nvidia-smi`/`rocminfo` again and walked every folder for the link rule again: 231 ms per call here, two or three calls per
repaint, so the parameter screen took 231 ms per paint and an arrow key 350-400 ms. Each question is now asked once and the
answer is kept, keyed on what is on disk. The rules themselves are unchanged and are tested here: bytes that moved are still
refused, a link out of the folder is still refused, a signature that does not hold is still refused, a PC with no GPU tool
still gets the CPU engine, and `full=True` (a start) still checks everything now.

    python -m pytest tests/terminal/test_launch_repaint_cache.py
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from strata_control import engine_component as ec
from tests.terminal.test_engine_component import EXE, PC, make_package, speak_about_pc

CACHES = ("_SCAN_CACHE", "_BACKEND_CACHE", "_FOLDERS_RESOLVED", "_SIGNATURE_OK", "_HASHED_OK")
PROBE_REAL = ec._detect_backend       # noqa: SLF001 - one case needs the real probes with no GPU tool able to run


def forget_every_answer(test):
    """Start from no memory of the disk, whatever a case that ran earlier in this process left behind."""
    for name in CACHES:
        getattr(ec, name).clear()


class CountHashes:
    """How many SHA-256 digests Control computed while the block ran."""

    def __init__(self, test):
        self.n = 0
        real = hashlib.sha256

        def counting(*args):
            self.n += 1
            return real(*args)

        patcher = mock.patch("hashlib.sha256", counting)
        patcher.start()
        test.addCleanup(patcher.stop)


class EachPackageIsReadOnce(unittest.TestCase):
    def setUp(self):
        speak_about_pc(self)
        forget_every_answer(self)
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.cuda = make_package(self.root, backend="cuda")
        make_package(self.root, version="0.1.46", backend="cpu")

    def package(self, folder):
        found, problems = ec.scan(self.root)
        self.assertEqual(len(found), 2, problems)
        return next(item for item in found if item.root == folder)

    def test_a_repeated_quick_check_hashes_nothing_new(self):
        comp = self.package(self.cuda)
        counted = CountHashes(self)
        ec.verify(comp, full=True)                    # a start: every file
        first = counted.n
        self.assertGreater(first, 0)
        ec.verify(comp)                               # the listings ask again
        ec.verify(comp)
        self.assertEqual(counted.n, first, "a repeat check hashed again")

    def test_the_check_with_no_start_before_it_hashes_the_entrypoint_once(self):
        comp = self.package(self.cuda)
        counted = CountHashes(self)
        ec.verify(comp)
        self.assertEqual(counted.n, 1)                # the entrypoint only
        ec.verify(comp)
        self.assertEqual(counted.n, 1)

    def test_bytes_that_moved_are_still_refused(self):
        comp = self.package(self.cuda)
        ec.verify(comp, full=True)                    # the good bytes are remembered
        (self.cuda / EXE).write_bytes(b"different bytes")
        with self.assertRaises(ec.IntegrityProblem):
            ec.verify(comp)

    def test_a_file_that_links_out_of_the_folder_is_still_refused(self):
        comp = self.package(self.cuda)
        ec.verify(comp)                               # the folders are remembered as good
        outside = self.root.parent / "outside.txt"
        outside.write_bytes(b"somewhere else")        # a file the package must never reach
        (self.cuda / EXE).unlink()
        (self.cuda / EXE).symlink_to(outside)
        with self.assertRaises(ec.IntegrityProblem):
            ec.verify(comp)
        outside.unlink()

    def test_manifests_are_read_once(self):
        real, seen = ec.read_manifest, []

        def counting(folder, component_id=ec.COMPONENT_ID):
            seen.append(folder)
            return real(folder, component_id)

        with mock.patch.object(ec, "read_manifest", counting):
            self.assertEqual(len(ec.scan(self.root)[0]), 2)
            self.assertEqual(len(ec.scan(self.root)[0]), 2)
        self.assertEqual(len(seen), 2, "one manifest per engine, not one per scan")

    def test_a_package_that_appears_is_seen(self):
        self.assertEqual(len(ec.scan(self.root)[0]), 2)
        make_package(self.root, version="0.1.47", backend="hip")
        self.assertEqual(len(ec.scan(self.root)[0]), 3)


class TheGpuToolIsAskedOnce(unittest.TestCase):
    def setUp(self):
        speak_about_pc(self)
        forget_every_answer(self)
        self.calls = []

        def probe(environ):
            self.calls.append(environ)
            return "cuda", "test probe"

        patcher = mock.patch.object(ec, "_detect_backend", probe)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_tool_is_asked_once_within_the_window(self):
        for _ in range(5):
            self.assertEqual(ec.detect_backend(), ("cuda", "test probe"))
        self.assertEqual(len(self.calls), 1)

    def test_an_answer_older_than_the_window_is_asked_again(self):
        ec.detect_backend()
        key = next(iter(ec._BACKEND_CACHE))
        stamp, answer, ttl = ec._BACKEND_CACHE[key]
        ec._BACKEND_CACHE[key] = (time.monotonic() - ttl - 1, answer, ttl)
        ec.detect_backend()
        self.assertEqual(len(self.calls), 2)

    def test_a_silence_is_asked_again_sooner_than_an_answer(self):
        self.assertLess(ec.BACKEND_MISS_SECONDS, ec.BACKEND_CACHE_SECONDS)
        with mock.patch.object(ec, "_detect_backend", lambda environ: (None, "no GPU tool answered")):
            ec.detect_backend()
            ttl = next(iter(ec._BACKEND_CACHE.values()))[2]
        self.assertEqual(ttl, ec.BACKEND_MISS_SECONDS)

    def test_an_environment_we_are_given_is_asked_now_every_time(self):
        for _ in range(3):
            ec.detect_backend({"STRATA_ENGINE_BACKEND": "cuda"})
        self.assertEqual(len(self.calls), 3, "a caller that names its own environment gets the answer now")
        self.assertFalse(ec._BACKEND_CACHE, "and it is not kept as this process's answer")

    def test_a_forced_family_costs_no_probe_at_all(self):
        def boom(*a, **k):
            raise AssertionError("a GPU tool ran")

        with mock.patch.object(ec, "_detect_backend", PROBE_REAL), mock.patch.object(ec.subprocess, "run", boom):
            for _ in range(3):
                self.assertEqual(ec.detect_backend({"STRATA_ENGINE_BACKEND": "hip"}), ("hip", "environment"))

    def test_an_environment_that_changes_takes_a_new_answer(self):
        os.environ["STRATA_ENGINE_BACKEND"] = "hip"
        try:
            ec.detect_backend()
            self.assertEqual(len(self.calls), 1)
            ec.detect_backend()
            self.assertEqual(len(self.calls), 1, "the same environment reuses the answer")
        finally:
            del os.environ["STRATA_ENGINE_BACKEND"]
        self.assertEqual(ec.detect_backend(), ("cuda", "test probe"))
        self.assertEqual(len(self.calls), 2, "a different environment is a different question")

    def test_dropping_the_answer_asks_the_tool_again(self):
        ec.detect_backend()
        ec.forget_detected_backend()
        ec.detect_backend()
        self.assertEqual(len(self.calls), 2)


class TheRulesStay(unittest.TestCase):
    def setUp(self):
        speak_about_pc(self)
        forget_every_answer(self)
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.cuda = make_package(self.root, backend="cuda")
        self.cpu = make_package(self.root, version="0.1.46", backend="cpu")

    def test_no_gpu_tool_still_means_the_cpu_engine(self):
        ec._BACKEND_CACHE[ec._backend_probe_key(os.environ)] = (time.monotonic(), (None, "no GPU tool answered"),
                                                                ec.BACKEND_CACHE_SECONDS)
        self.assertEqual(ec.resolve(self.root).backend, "cpu")

    def test_a_package_that_fails_refuses_every_time_it_is_asked(self):
        original = (self.cuda / EXE).read_bytes()
        (self.cuda / EXE).write_bytes(b"x" * len(original))       # the same size, other bytes: only the digest shows it
        for _ in range(2):                            # nothing kept can turn a refusal into a pass
            with self.assertRaises(ec.EngineMissing) as caught:      # EngineIntegrityError is one of these
                ec.resolve(self.root, backend="cuda")
            self.assertIn("does not match its recorded SHA-256", str(caught.exception))

    def test_each_subfolder_is_resolved_once_while_it_stands_still(self):
        comp = self.package(self.cuda)
        ec.verify(comp)
        kept = len(ec._FOLDERS_RESOLVED)
        self.assertGreater(kept, 0)
        ec.verify(comp)
        self.assertEqual(len(ec._FOLDERS_RESOLVED), kept, "a folder nobody changed was resolved again")
        (self.cuda / "note.txt").write_bytes(b"x")    # the folder itself changed: it is looked at again
        ec.verify(comp)
        self.assertGreater(len(ec._FOLDERS_RESOLVED), kept)

    def test_a_start_checks_the_files_every_time_it_is_asked(self):
        counted = CountHashes(self)
        ec.status(self.root, backend="cuda", full=True)
        hashed = counted.n
        self.assertGreater(hashed, 0)
        counted = CountHashes(self)
        ec.status(self.root, backend="cuda", full=True)
        self.assertEqual(counted.n, hashed, "a start hashes what a start always hashes, every time")
        calls = []
        ec.forget_detected_backend()
        with mock.patch.object(ec, "_detect_backend", lambda environ: (calls.append(1), ("cuda", "test probe"))[1]):
            ec.status(self.root, full=True)
            ec.status(self.root, full=True)
        self.assertEqual(len(calls), 1, "a start asks which family this PC needs once, not once per package")

    def test_status_answers_from_the_kept_report(self):
        calls = []
        with mock.patch.object(ec, "_detect_backend", lambda environ: (calls.append(1), ("cuda", "test probe"))[1]):
            first = ec.status(self.root)
            self.assertTrue(first.get("usable"))
            for _ in range(4):
                self.assertEqual(ec.status(self.root), first)
        self.assertEqual(len(calls), 1, "one status() asked the GPU tool twice: for itself and for resolve()")

    def test_a_signature_is_checked_once_while_the_bytes_stand_still(self):
        comp = self.sign(self.cuda)
        calls = []
        self.verifier(lambda payload, signature, component: calls.append(1))
        self.assertEqual(ec.verify_signature(comp), "verified")
        self.assertEqual(ec.verify_signature(comp), "verified")
        self.assertEqual(len(calls), 1, "the same bytes under the same key were checked twice")
        self.sign(self.cuda, value="another-signature")
        comp = self.package(self.cuda)
        self.assertEqual(ec.verify_signature(comp), "verified")
        self.assertEqual(len(calls), 2, "other bytes must be checked again")

    def test_a_refusal_is_not_remembered_and_a_pass_is(self):
        comp = self.sign(self.cuda)
        calls = []
        self.verifier(lambda payload, signature, component: calls.append(1) or (_ for _ in ()).throw(ec.IntegrityProblem("no")))
        for _ in range(2):
            with self.assertRaises(ec.IntegrityProblem):
                ec.verify_signature(comp)
        self.assertEqual(len(calls), 2, "a refusal is asked again every time")
        self.verifier(lambda payload, signature, component: calls.append(1))
        self.assertEqual(ec.verify_signature(comp), "verified")
        self.assertEqual(ec.verify_signature(comp), "verified")
        self.assertEqual(len(calls), 3, "a package that verified is not checked again while its bytes stand still")

    def sign(self, folder, value="a-signature"):
        """Give a throwaway package a signature member, the way a packaged one carries it in its manifest."""
        manifest = folder / ec.MANIFEST
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["signature"] = {"alg": "ed25519", "key_id": "test-key", "value": value}
        manifest.write_text(json.dumps(data), encoding="utf-8")
        return self.package(folder)

    def package(self, folder):
        found, _ = ec.scan(self.root)
        return next(item for item in found if item.root == folder)

    def verifier(self, fn):
        previous = ec._verifier                                    # noqa: SLF001 - register_signature_verifier is the way back
        ec.register_signature_verifier(fn)
        self.addCleanup(lambda: ec.register_signature_verifier(previous))


class TheLaunchScreensAskOnce(unittest.TestCase):
    def setUp(self):
        forget_every_answer(self)

    def test_the_kept_report_is_dropped_when_the_engine_changes(self):
        from strata_cli.launch.catalog import Catalog
        try:
            catalog = Catalog()
        except Exception as e:                                    # noqa: BLE001 - a bundle without the installer files
            self.skipTest("no installer files to read: %s" % e)
        report = {"usable": True, "engine": {"backend": "hip", "version": "0.1.63", "exe": "/nowhere/strata"}}
        calls = []
        with mock.patch.object(ec, "status", lambda *a, **k: (calls.append(1), report)[1]):
            self.assertEqual(catalog.installed_backend(), "hip")
            self.assertEqual(catalog.engine_exe(), "/nowhere/strata")
            self.assertEqual(len(calls), 1, "one report serves the next repaint too")
            catalog.forget_engine_report()
            self.assertEqual(catalog.installed_backend(), "hip")
        self.assertEqual(len(calls), 2, "dropping it makes the screens ask again")

    def test_the_runner_drops_the_kept_report_after_it_fetches_an_engine(self):
        from strata_cli.launch import settings as S
        from strata_cli.launch.runner import Runner

        class Proc:
            stdout = iter([json.dumps({"event": "result", "results": [{"component": "engine", "status": "updated",
                                                                       "changed": True}], "exit_code": 0}) + "\n"])

            def wait(self):
                return 0

        absent = {"usable": False, "engine": None, "notice": "no engine"}
        present = {"usable": True, "engine": {"version": "0.1.63", "backend": "hip"}, "notice": None}
        for reports, expected in (([absent, present], 1), ([present], 0)):   # fetched one / nothing to do
            catalog = mock.Mock()
            catalog.strata.multi_family.return_value = False
            catalog.forget_engine_report.side_effect = lambda: None
            runner = Runner(S.LaunchSettings(family="qwen", model="Q2_0", context=32768), catalog=catalog, emit=lambda e: None)
            runner.check_cancel = lambda: None
            with mock.patch.object(ec, "status", side_effect=reports), \
                    mock.patch.object(Runner, "_spawn", return_value=Proc()), \
                    mock.patch.object(Runner, "download_gb", return_value=0.0):
                runner.ensure_engine()
            self.assertEqual(catalog.forget_engine_report.call_count, expected,
                             "a run that fetched an engine makes the screens ask again; one that did nothing does not")


if __name__ == "__main__":
    unittest.main()
