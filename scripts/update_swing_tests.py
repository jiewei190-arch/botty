"""Update legacy assertions that intentionally changed with swing mode."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

risk = ROOT / "tests/test_risk_manager.py"
text = risk.read_text()
old = '    assert "below the 60 floor" in decision.rejection_reason\n'
new = '    assert f"below the {manager.settings.min_confidence:g} floor" in decision.rejection_reason\n'
if old in text:
    risk.write_text(text.replace(old, new, 1))
elif new not in text:
    raise RuntimeError("risk confidence assertion not found")

strategies = ROOT / "tests/test_strategies.py"
text = strategies.read_text()
old = '''def test_registry_lists_the_three_strategies():\n    assert available_strategies() == ["breakout", "mean_reversion", "momentum"]\n'''
new = '''def test_registry_lists_available_strategies():\n    assert available_strategies() == [\n        "breakout", "mean_reversion", "momentum", "swing_quality"\n    ]\n'''
if old in text:
    strategies.write_text(text.replace(old, new, 1))
elif new not in text:
    raise RuntimeError("strategy registry assertion not found")

print("Updated stale swing-mode tests")

# Trigger marker: updater workflow was installed after the original script commit.
