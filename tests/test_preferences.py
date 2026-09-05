"""Dashboard preferences: the remembered account balance.

The balance is entered by hand, because the data provider and the broker you
trade need not be the same place. Remembering the last entry removes the
retyping without ever becoming a source of truth — the number in the box is
what sizing uses.
"""

from __future__ import annotations

import json

import pytest

from trading_bot.dashboard.preferences import (
    PREFERENCES_FILE,
    load_preferences,
    remembered_equity,
    save_preference,
)


class TestRoundTrip:
    def test_a_saved_value_comes_back(self, tmp_path):
        save_preference(tmp_path, "account_equity", 15_000.0)
        assert remembered_equity(tmp_path, 10_000.0) == 15_000.0

    def test_it_survives_a_restart(self, tmp_path):
        """Written to disk, not held in memory."""
        save_preference(tmp_path, "account_equity", 22_500.0)
        assert json.loads((tmp_path / PREFERENCES_FILE).read_text()) == {
            "account_equity": 22_500.0
        }

    def test_saving_one_key_leaves_the_others(self, tmp_path):
        save_preference(tmp_path, "account_equity", 15_000.0)
        save_preference(tmp_path, "theme", "dark")
        stored = load_preferences(tmp_path)
        assert stored == {"account_equity": 15_000.0, "theme": "dark"}

    def test_an_unchanged_value_does_not_rewrite(self, tmp_path):
        save_preference(tmp_path, "account_equity", 15_000.0)
        stamp = (tmp_path / PREFERENCES_FILE).stat().st_mtime_ns
        save_preference(tmp_path, "account_equity", 15_000.0)
        assert (tmp_path / PREFERENCES_FILE).stat().st_mtime_ns == stamp


class TestFallbacks:
    """A preference file is a convenience. It must never stop the app."""

    def test_no_file_falls_back(self, tmp_path):
        assert remembered_equity(tmp_path, 10_000.0) == 10_000.0
        assert load_preferences(tmp_path) == {}

    def test_corrupt_json_falls_back(self, tmp_path):
        (tmp_path / PREFERENCES_FILE).write_text("{not json")
        assert remembered_equity(tmp_path, 10_000.0) == 10_000.0

    def test_a_non_object_file_falls_back(self, tmp_path):
        (tmp_path / PREFERENCES_FILE).write_text('["a", "list"]')
        assert load_preferences(tmp_path) == {}

    @pytest.mark.parametrize("bad", ["abc", None, "", -5, 0])
    def test_an_unusable_equity_falls_back(self, tmp_path, bad):
        """A zero or negative balance would divide sizing by nothing."""
        (tmp_path / PREFERENCES_FILE).write_text(
            json.dumps({"account_equity": bad})
        )
        assert remembered_equity(tmp_path, 10_000.0) == 10_000.0

    def test_an_unwritable_directory_is_survivable(self, tmp_path):
        target = tmp_path / "file-not-a-dir"
        target.write_text("blocked")
        save_preference(target / "nested", "account_equity", 15_000.0)  # must not raise
