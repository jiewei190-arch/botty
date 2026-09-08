"""One-time branch patch: make swing_quality the CLI default for hunt/backtest."""

from pathlib import Path

path = Path(__file__).resolve().parents[1] / "trading_bot/main.py"
text = path.read_text()

old_hunt = '''    hunt.add_argument(\n        "--strategy", default="all",\n        help="Strategy name, comma-separated list, or 'all' (default). "\n'''
new_hunt = '''    hunt.add_argument(\n        "--strategy", default="swing_quality",\n        help="Strategy name, comma-separated list, or 'all' (default: swing_quality). "\n'''
if text.count(old_hunt) != 1:
    raise RuntimeError(f"hunt default match count: {text.count(old_hunt)}")
text = text.replace(old_hunt, new_hunt, 1)

old_backtest = '''    backtest.add_argument(\n        "--strategy",\n        default="momentum",\n        help="Strategy name, comma-separated list, or 'all'. "\n'''
new_backtest = '''    backtest.add_argument(\n        "--strategy",\n        default="swing_quality",\n        help="Strategy name, comma-separated list, or 'all' (default: swing_quality). "\n'''
if text.count(old_backtest) != 1:
    raise RuntimeError(f"backtest default match count: {text.count(old_backtest)}")
text = text.replace(old_backtest, new_backtest, 1)

path.write_text(text)
print("Patched swing CLI defaults")
