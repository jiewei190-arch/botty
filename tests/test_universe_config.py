"""The categorised symbol universe."""

from __future__ import annotations

import json

import pytest

from trading_bot.config import universe as uni


def test_built_in_categories_are_normalised():
    catalogue = uni.default_catalogue()
    for name, symbols in catalogue.categories.items():
        assert name == name.upper()
        assert all(symbol == symbol.upper().strip() for symbol in symbols)
        assert len(set(symbols)) == len(symbols), f"{name} has duplicates"


def test_default_stream_set_fits_the_free_plan():
    """The shipped default must not fail on the plan most users have."""
    catalogue = uni.default_catalogue()
    symbols = catalogue.resolve(uni.DEFAULT_STREAM_CATEGORIES)
    assert len(symbols) <= uni.FREE_STREAM_SYMBOL_LIMIT
    assert uni.stream_capacity_error(symbols) is None


def test_resolve_deduplicates_across_categories_and_keeps_order():
    catalogue = uni.default_catalogue()
    symbols = catalogue.resolve(["MEGA_CAPS", "TECH"])
    assert symbols[0] == "AAPL"
    assert len(set(symbols)) == len(symbols)


def test_resolve_applies_includes_and_excludes():
    catalogue = uni.default_catalogue()
    symbols = catalogue.resolve(["INDEX_ETFS"], include=["pltr"], exclude=["spy"])
    assert "PLTR" in symbols
    assert "SPY" not in symbols


def test_unknown_category_names_the_available_ones():
    with pytest.raises(uni.UniverseConfigError, match="INDEX_ETFS"):
        uni.default_catalogue().category("NOT_A_CATEGORY")


def test_capacity_error_explains_the_overflow():
    message = uni.stream_capacity_error([f"S{i}" for i in range(45)])
    assert message is not None
    assert "45 symbols" in message and "30" in message


def test_capacity_error_counts_unique_symbols():
    assert uni.stream_capacity_error(["AAPL"] * 100, limit=5) is None


class TestFileOverrides:
    def test_missing_file_falls_back_to_built_ins(self, tmp_path):
        catalogue = uni.load_catalogue(tmp_path / "nope.json")
        assert catalogue.categories == uni.default_catalogue().categories

    def test_file_replaces_a_category_rather_than_merging(self, tmp_path):
        path = tmp_path / "universe.json"
        path.write_text(json.dumps({"ENERGY": ["xom"]}))
        catalogue = uni.load_catalogue(path)
        # Replacement, not a merge: shrinking a category has to be possible.
        assert catalogue.category("ENERGY") == ("XOM",)
        assert catalogue.category("INDEX_ETFS") == uni.INDEX_ETFS

    def test_empty_list_removes_a_category(self, tmp_path):
        path = tmp_path / "universe.json"
        path.write_text(json.dumps({"ENERGY": []}))
        assert "ENERGY" not in uni.load_catalogue(path).names

    def test_new_category_is_added(self, tmp_path):
        path = tmp_path / "universe.json"
        path.write_text(json.dumps({"my_list": ["aapl", "AAPL", " nvda "]}))
        assert uni.load_catalogue(path).category("MY_LIST") == ("AAPL", "NVDA")

    def test_malformed_json_raises_rather_than_silently_reverting(self, tmp_path):
        path = tmp_path / "universe.json"
        path.write_text("{not json")
        with pytest.raises(uni.UniverseConfigError, match="Could not read"):
            uni.load_catalogue(path)

    def test_non_list_category_is_rejected(self, tmp_path):
        path = tmp_path / "universe.json"
        path.write_text(json.dumps({"ENERGY": "XOM"}))
        with pytest.raises(uni.UniverseConfigError, match="must be a list"):
            uni.load_catalogue(path)
