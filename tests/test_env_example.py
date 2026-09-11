"""The shipped `.env.example` must satisfy the settings it configures.

A deployment copies this file to become its real environment file, so a value
the code rejects is not a documentation slip — it is a container that will not
boot. That happened: the option DTE window was derived from the holding window
in code and a validator added to enforce the relationship, but the template
kept the old numbers. Every fresh install would have copied
``OPTIONS_MIN_DTE=7`` alongside ``OPTIONS_PLANNED_MAX_HOLD_DAYS=60`` and died
on start with a configuration error, and every existing install inherited the
same values the moment the new code ran.

Nothing pinned the template to the code, so nothing noticed. This does.
"""

from __future__ import annotations

import pathlib

import pytest

from trading_bot.config.settings import Settings

ENV_EXAMPLE = pathlib.Path(__file__).resolve().parent.parent / ".env.example"


def parse_env_example() -> dict[str, str]:
    """The assignments in `.env.example`, ignoring comments and blanks."""
    values: dict[str, str] = {}
    for raw in ENV_EXAMPLE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def test_the_template_exists_and_is_not_empty():
    assert ENV_EXAMPLE.is_file()
    assert parse_env_example()


def test_the_shipped_template_loads_through_every_validator(monkeypatch):
    """The regression: this raised on the option DTE window.

    Constructed rather than merely parsed, because the defect was a *relation*
    between four values that each looked reasonable alone.
    """
    for key, value in parse_env_example().items():
        monkeypatch.setenv(key, value)

    Settings()  # must not raise


def test_a_template_that_contradicts_the_validator_is_caught(monkeypatch):
    """Proof the test above can fail — the exact combination that shipped."""
    for key, value in parse_env_example().items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("OPTIONS_MIN_DTE", "7")
    monkeypatch.setenv("OPTIONS_PLANNED_MAX_HOLD_DAYS", "60")

    with pytest.raises(Exception, match="MIN_DTE"):
        Settings()


def test_the_documented_expiry_window_outlives_the_documented_hold():
    """The relationship in prose form, read straight from the file."""
    env = parse_env_example()
    min_dte = int(env["OPTIONS_MIN_DTE"])
    hold = int(env["OPTIONS_PLANNED_MAX_HOLD_DAYS"])
    buffer_days = int(env["OPTIONS_EXIT_BEFORE_EXPIRY_DAYS"])

    assert min_dte >= hold + buffer_days


def test_the_template_documents_every_option_setting():
    """A setting absent here is undiscoverable: the installer never rewrites
    an existing environment file, so a new key reaches an operator only by
    being visible in this template."""
    documented = {
        key[len("OPTIONS_"):].lower()
        for key in parse_env_example()
        if key.startswith("OPTIONS_")
    }
    from trading_bot.config.settings import OptionsSettings

    missing = set(OptionsSettings.model_fields) - documented
    # Recorded rather than asserted empty: the moneyness and liquidity floors
    # are deliberately undocumented knobs. Anything newly missing shows here.
    # The contract-quality floors are deliberately left out: they describe
    # what counts as a tradeable contract rather than what the account risks,
    # and an operator who needs them can read the settings module. Anything
    # newly missing that is not one of these shows up here.
    contract_quality_floors = {
        "target_delta", "min_abs_delta", "max_abs_delta", "max_spread_pct",
        "min_daily_volume", "min_open_interest",
        "fallback_min_moneyness_pct", "fallback_max_moneyness_pct",
    }
    assert missing <= contract_quality_floors, (
        f"new undocumented option settings: {sorted(missing - contract_quality_floors)}"
    )
