"""lillypad.paths — the one module that resolves a location.

These tests exist because this module's correctness is invisible at runtime until
it is wrong: a `parents[N]` counted one level off does not raise, it just makes
every icon fail to load and the calibration menu come up empty. They also pin the
frozen branches, which are otherwise only exercised by building the exe.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lillypad import paths


class TestFromSource:
    def test_project_root_is_the_repository_root(self, repo_root):
        """paths.py sits one level down, in lillypad/, so parents[1] is the root.
        If this is off by one, nothing raises — every asset just goes missing."""
        assert paths._PROJECT_ROOT == repo_root

    def test_app_dir_is_the_repository_root(self, repo_root):
        assert paths.app_dir() == repo_root

    def test_resource_path_with_no_parts_is_the_base_directory(self, repo_root):
        """The Avantes DLL search calls it this way."""
        assert paths.resource_path() == repo_root

    def test_resource_path_joins_parts(self, repo_root):
        assert paths.resource_path("icons", "x.png") == repo_root / "icons" / "x.png"

    def test_the_bundled_assets_the_app_actually_asks_for_are_there(self):
        """A smoke test over the real repo: these are the two datas entries in
        Lillypad.spec, and both are referenced by relative name."""
        assert paths.resource_path("icons").is_dir()
        assert paths.resource_path("icons", "Lilypad.ico").is_file()
        assert paths.resource_path("calibration_files").is_dir()

    def test_settings_and_calibration_sit_beside_the_program(self, repo_root):
        assert paths.CALIBRATION_DIR == repo_root / "calibration_files"
        assert paths.SETTINGS_PATH == repo_root / "settings.json"


class TestFrozen:
    """The two functions diverge when frozen, and the ordering is load-bearing:
    each tests its own sys attribute FIRST, so neither ever falls back to
    _PROJECT_ROOT — which inside a PYZ archive would evaluate to the bundle
    directory rather than anything useful.
    """

    def test_resource_path_prefers_meipass(self, monkeypatch, tmp_path):
        monkeypatch.setattr(paths.sys, "_MEIPASS", str(tmp_path), raising=False)
        assert paths.resource_path("icons") == tmp_path / "icons"

    def test_resource_path_ignores_project_root_when_bundled(self, monkeypatch,
                                                             tmp_path, repo_root):
        monkeypatch.setattr(paths.sys, "_MEIPASS", str(tmp_path), raising=False)
        assert repo_root not in paths.resource_path("icons").parents

    def test_app_dir_is_beside_the_executable_not_in_the_bundle(
            self, monkeypatch, tmp_path):
        """The distinction the module exists for: the operator's files go next to
        the .exe, because _MEIPASS is re-extracted on every launch."""
        exe = tmp_path / "install" / "Lillypad.exe"
        exe.parent.mkdir()
        exe.touch()
        monkeypatch.setattr(paths.sys, "frozen", True, raising=False)
        monkeypatch.setattr(paths.sys, "executable", str(exe))
        monkeypatch.setattr(paths.sys, "_MEIPASS", str(tmp_path / "bundle"),
                            raising=False)
        assert paths.app_dir() == exe.parent
        assert paths.app_dir() != paths.resource_path()


class TestSeedCalibrationDir:
    """First-run seeding for the frozen build. `monkeypatch.setattr` on the
    module constants is how the real frozen layout is simulated: the function
    reads them at call time."""

    @staticmethod
    def _stage(monkeypatch, tmp_path, *, names=("A.txt", "B.txt")):
        bundled = tmp_path / "bundle" / "calibration_files"
        bundled.mkdir(parents=True)
        for n in names:
            (bundled / n).write_text("400 1.0\n500 1.0\n")
        (bundled / "notes.md").write_text("not a calibration file")
        target = tmp_path / "install" / "calibration_files"
        target.parent.mkdir()
        monkeypatch.setattr(paths, "CALIBRATION_DIR", target)
        monkeypatch.setattr(paths, "resource_path",
                            lambda *p: tmp_path / "bundle" / Path(*p))
        return bundled, target

    def test_copies_the_bundled_txt_files_on_first_run(self, monkeypatch, tmp_path):
        _, target = self._stage(monkeypatch, tmp_path)
        paths.seed_calibration_dir()
        assert sorted(f.name for f in target.iterdir()) == ["A.txt", "B.txt"]

    def test_copies_only_txt_files(self, monkeypatch, tmp_path):
        _, target = self._stage(monkeypatch, tmp_path)
        paths.seed_calibration_dir()
        assert not (target / "notes.md").exists()

    def test_never_touches_a_folder_the_user_already_owns(self, monkeypatch,
                                                          tmp_path):
        """The operator's own calibration files must survive every launch, and a
        file they deleted must stay deleted."""
        bundled, target = self._stage(monkeypatch, tmp_path)
        target.mkdir()
        (target / "mine.txt").write_text("600 2.0\n700 2.0\n")
        paths.seed_calibration_dir()
        assert sorted(f.name for f in target.iterdir()) == ["mine.txt"]

    def test_is_a_no_op_when_source_and_target_are_the_same(self, monkeypatch,
                                                            tmp_path):
        """Running from source, the bundle IS the folder beside the program —
        seeding would be copying files onto themselves."""
        same = tmp_path / "calibration_files"
        same.mkdir()
        (same / "A.txt").write_text("400 1.0\n500 1.0\n")
        monkeypatch.setattr(paths, "CALIBRATION_DIR", same)
        monkeypatch.setattr(paths, "resource_path", lambda *p: tmp_path / Path(*p))
        # The guard is the `exists()` check; assert it does not raise or duplicate.
        paths.seed_calibration_dir()
        assert sorted(f.name for f in same.iterdir()) == ["A.txt"]

    def test_a_missing_bundle_is_not_an_error(self, monkeypatch, tmp_path):
        target = tmp_path / "install" / "calibration_files"
        target.parent.mkdir()
        monkeypatch.setattr(paths, "CALIBRATION_DIR", target)
        monkeypatch.setattr(paths, "resource_path",
                            lambda *p: tmp_path / "nothing-here" / Path(*p))
        paths.seed_calibration_dir()
        assert not target.exists()


class TestAvantesSearchDirInjection:
    """avantes.py must not have to work out where Lillypad is installed, so the
    adapter tells it. This is what replaced the vendor module's own __file__
    anchor as the load-bearing path."""

    def test_add_search_dir_puts_the_directory_ahead_of_the_install_folders(self):
        import avantes
        probe = Path(r"C:\some\program\folder")
        try:
            avantes.add_search_dir(probe)
            candidates = avantes._candidate_paths()
            ours = [i for i, p in enumerate(candidates) if str(probe) in p]
            assert ours, "the registered directory was not searched at all"
            versioned = [i for i, p in enumerate(candidates)
                         if "AvaSpec" in p and str(probe) not in p]
            if versioned:
                assert min(ours) < min(versioned)
        finally:
            avantes._extra_dirs.clear()

    def test_add_search_dir_is_idempotent(self):
        import avantes
        try:
            avantes.add_search_dir(r"C:\x")
            avantes.add_search_dir(r"C:\x")
            assert avantes._extra_dirs == [str(Path(r"C:\x"))]
        finally:
            avantes._extra_dirs.clear()

    def test_the_env_override_still_outranks_it(self, monkeypatch):
        import avantes
        monkeypatch.setenv(avantes._ENV_VAR, r"C:\mine\AvaSpecX64.dll")
        try:
            avantes.add_search_dir(r"C:\program")
            assert avantes._candidate_paths()[0] == r"C:\mine\AvaSpecX64.dll"
        finally:
            avantes._extra_dirs.clear()

    def test_the_adapter_registers_the_program_folder(self, monkeypatch):
        """hardware._avantes_module() is the single place that does it, so every
        Avantes entry point gets it without having to remember."""
        import avantes
        import hardware
        try:
            avantes._extra_dirs.clear()
            hardware._avantes_module()
            assert str(paths.app_dir()) in avantes._extra_dirs
        finally:
            avantes._extra_dirs.clear()
