"""lillypad.settings — reading and writing settings.json.

New tests. The write path is worth pinning precisely because its failure mode is
quiet and asymmetric: a torn file loses the PREVIOUS session's settings too,
which is strictly worse than not saving at all. That is why it writes to a temp
file and renames, and why a failure is swallowed rather than raised.
"""

from __future__ import annotations

import json

import pytest

from lillypad import settings


@pytest.fixture
def settings_file(monkeypatch, tmp_path):
    """Point the module at a throwaway settings.json and clear its cache."""
    path = tmp_path / "settings.json"
    monkeypatch.setattr(settings, "SETTINGS_PATH", path)
    settings._reset_cache()
    yield path
    settings._reset_cache()


def _over_defaults(**file_values):
    """What load_settings should return for a file holding `file_values`."""
    return {**settings.DEFAULT_SETTINGS, **file_values}


class TestLoad:
    def test_a_missing_file_is_the_ordinary_first_run_case(self, settings_file):
        """A fresh install starts on the defaults."""
        assert settings.load_settings() == settings.DEFAULT_SETTINGS

    def test_reads_a_good_file(self, settings_file):
        settings_file.write_text(json.dumps({"theme": "light", "cmap": "jet"}),
                                 encoding="utf-8")
        assert settings.load_settings() == _over_defaults(theme="light",
                                                          cmap="jet")

    def test_the_file_wins_over_every_default(self, settings_file):
        settings_file.write_text(json.dumps({"saturation_pct": 97.0,
                                             "export_name": "x"}),
                                 encoding="utf-8")
        got = settings.load_settings()
        assert got["saturation_pct"] == 97.0 and got["export_name"] == "x"

    def test_the_defaults_survive_a_caller_editing_what_it_loaded(
            self, settings_file):
        """The nested window dict is copied, never shared."""
        settings.load_settings()["window"]["w"] = 1
        settings._reset_cache()
        assert settings.load_settings()["window"] == \
            settings.DEFAULT_SETTINGS["window"]
        assert settings.DEFAULT_SETTINGS["window"]["w"] != 1

    @pytest.mark.parametrize("content,label", [
        ('{"theme": "light"', "truncated"),
        ("not json at all", "garbage"),
        ("[1, 2, 3]", "a list rather than an object"),
        ("null", "a bare null"),
    ])
    def test_anything_wrong_with_it_falls_back_to_defaults(self, settings_file,
                                                           content, label):
        """A startup that refused to start because a preferences file lost a
        brace would be a far worse failure than a forgotten colormap."""
        settings_file.write_text(content, encoding="utf-8")
        assert settings.load_settings() == settings.DEFAULT_SETTINGS, label

    def test_is_cached_for_the_process(self, settings_file):
        """main() needs the theme before the QApplication is styled and the
        window needs everything else afterwards; the two must not disagree."""
        settings_file.write_text('{"theme": "light"}', encoding="utf-8")
        first = settings.load_settings()
        settings_file.write_text('{"theme": "dark"}', encoding="utf-8")
        assert settings.load_settings() is first
        assert settings.load_settings()["theme"] == "light"


class TestSave:
    def test_round_trips(self, settings_file):
        assert settings.save_settings({"theme": "dark", "spec_frac": 0.5})
        assert json.loads(settings_file.read_text(encoding="utf-8")) == {
            "theme": "dark", "spec_frac": 0.5}

    def test_leaves_no_temp_file_behind(self, settings_file):
        settings.save_settings({"theme": "dark"})
        assert not settings_file.with_name(settings_file.name + ".tmp").exists()

    def test_numpy_scalars_do_not_abort_the_write(self, settings_file):
        """matplotlib hands back numpy scalars from get_ylim() and friends, and
        one of them must not cost the whole file."""
        np = pytest.importorskip("numpy")
        assert settings.save_settings({"line_width": np.float64(1.5),
                                       "idle_shots": np.int64(3)})
        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert data == {"line_width": 1.5, "idle_shots": 3.0}

    def test_an_unserialisable_value_fails_without_destroying_the_old_file(
            self, settings_file):
        """The reason for the temp file: the previous session's settings survive
        a failed write."""
        settings_file.write_text('{"theme": "light"}', encoding="utf-8")
        assert settings.save_settings({"bad": object()}) is False
        assert json.loads(settings_file.read_text(encoding="utf-8")) == {
            "theme": "light"}
        assert not settings_file.with_name(settings_file.name + ".tmp").exists()

    def test_an_unwritable_location_fails_silently(self, monkeypatch, tmp_path):
        """A read-only install directory should cost the operator nothing but the
        memory of their preferences."""
        monkeypatch.setattr(settings, "SETTINGS_PATH",
                            tmp_path / "no-such-dir" / "settings.json")
        assert settings.save_settings({"theme": "dark"}) is False

    def test_a_saved_file_reloads_after_a_cache_reset(self, settings_file):
        settings.save_settings({"theme": "light", "cmap": "viridis"})
        settings._reset_cache()
        assert settings.load_settings() == _over_defaults(theme="light",
                                                          cmap="viridis")


class TestAsBool:
    @pytest.mark.parametrize("v", [True, False])
    def test_passes_real_booleans_through(self, v):
        assert settings.as_bool(v) is v

    @pytest.mark.parametrize("v", ["false", "true", "", 0, 1, None, [], {}])
    def test_rejects_everything_else(self, v):
        """bool("false") is True, so a hand-edited file would otherwise mean the
        exact opposite of what it says. The caller catches this and falls back to
        that key's default."""
        with pytest.raises(TypeError):
            settings.as_bool(v)


def test_settings_version_is_an_int():
    assert isinstance(settings.SETTINGS_VERSION, int)
