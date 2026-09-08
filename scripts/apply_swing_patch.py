"""One-time deterministic patch for swing backtest diagnostics/UI wiring.

This script is intentionally strict: every replacement must match exactly or it
raises instead of partially editing a trading system. It is run once by a branch-
scoped GitHub Action and never touches main.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def patch_engine() -> None:
    path = ROOT / "trading_bot/backtesting/engine.py"
    text = path.read_text()

    text = replace_once(
        text,
        "    SignalDirection,\n)",
        "    SignalDirection,\n    explain_blockers,\n)",
        "engine import explain_blockers",
    )

    text = replace_once(
        text,
        "    rejection_reasons: dict[str, int] = field(default_factory=dict)\n",
        "    rejection_reasons: dict[str, int] = field(default_factory=dict)\n"
        "    strategy_blockers: dict[str, int] = field(default_factory=dict)\n"
        "    evaluations: int = 0\n",
        "BacktestResult diagnostic fields",
    )

    text = replace_once(
        text,
        "        self._rejections: dict[str, int] = {}\n",
        "        self._rejections: dict[str, int] = {}\n"
        "        self._strategy_blockers: dict[str, int] = {}\n"
        "        self._evaluations = 0\n",
        "reset diagnostics",
    )

    text = replace_once(
        text,
        "            rejection_reasons=self._rejections,\n        )",
        "            rejection_reasons=self._rejections,\n"
        "            strategy_blockers=self._strategy_blockers,\n"
        "            evaluations=self._evaluations,\n"
        "        )",
        "result diagnostics",
    )

    old_generate = '''            for strategy in self.strategies:\n                try:\n                    signal = strategy.generate_signal(symbol, history)\n                except Exception:  # noqa: BLE001 - one symbol must not stop the run\n                    logger.exception("%s failed on %s at %s", strategy.name, symbol, stamp)\n                    continue\n                if signal is not None:\n                    self._signals += 1\n                    self._pending.append((symbol, signal))\n                    break  # one position per symbol; first strategy wins\n'''
    new_generate = '''            for strategy in self.strategies:\n                self._evaluations += 1\n                try:\n                    signal = strategy.generate_signal(symbol, history)\n                except Exception:  # noqa: BLE001 - one symbol must not stop the run\n                    logger.exception("%s failed on %s at %s", strategy.name, symbol, stamp)\n                    key = f"{strategy.name}: evaluation_error"\n                    self._strategy_blockers[key] = self._strategy_blockers.get(key, 0) + 1\n                    continue\n                if signal is not None:\n                    self._signals += 1\n                    self._pending.append((symbol, signal))\n                    break  # one position per symbol; first strategy wins\n\n                blockers = explain_blockers(strategy)\n                if blockers:\n                    for blocker in blockers:\n                        key = f"{strategy.name}: {blocker}"\n                        self._strategy_blockers[key] = self._strategy_blockers.get(key, 0) + 1\n                elif strategy.last_evaluation:\n                    # Required conditions passed but generate_signal still returned\n                    # None: the confidence floor is the usual remaining gate.\n                    key = f"{strategy.name}: confidence_floor"\n                    self._strategy_blockers[key] = self._strategy_blockers.get(key, 0) + 1\n'''
    text = replace_once(text, old_generate, new_generate, "generate diagnostics")

    text = replace_once(
        text,
        '            "rejection_reasons": dict(self.rejection_reasons),\n            "trades": self.trades,\n',
        '            "rejection_reasons": dict(self.rejection_reasons),\n'
        '            "strategy_blockers": dict(self.strategy_blockers),\n'
        '            "evaluations": self.evaluations,\n'
        '            "trades": self.trades,\n',
        "result serialization diagnostics",
    )

    path.write_text(text)


def patch_dashboard() -> None:
    path = ROOT / "trading_bot/dashboard/app.py"
    text = path.read_text()

    text = replace_once(
        text,
        '        min_confidence = st.slider("Minimum confidence", 0, 100, 55, step=5)\n',
        '        min_confidence = st.slider(\n'
        '            "Minimum confidence", 0, 100, int(settings.risk.min_confidence), step=5\n'
        '        )\n',
        "dashboard confidence default",
    )

    text = replace_once(
        text,
        '            "Strategies", available_strategies(), default=["momentum"],\n',
        '            "Strategies", available_strategies(), default=["swing_quality"],\n',
        "backtest default strategy",
    )

    old_request_tail = '''            risk=settings.risk.model_copy(\n                update={\n                    "max_risk_per_trade_pct": float(risk_pct),\n                    "max_open_positions": int(max_positions),\n                }\n            ),\n            demo=controls["demo"],\n'''
    new_request_tail = '''            risk=settings.risk.model_copy(\n                update={\n                    "max_risk_per_trade_pct": float(risk_pct),\n                    "max_open_positions": int(max_positions),\n                    "min_confidence": float(controls["min_confidence"]),\n                }\n            ),\n            allow_short=bool(controls["allow_short"]),\n            min_confidence=float(controls["min_confidence"]),\n            demo=controls["demo"],\n'''
    text = replace_once(text, old_request_tail, new_request_tail, "backtest request controls")

    text = replace_once(
        text,
        '''        st.info(\n            "No trades were taken. Widen the date range, lower the confidence "\n            "floor on the Strategy Settings page, or try another strategy."\n        )\n''',
        '''        st.info(\n            "No trades were taken. That can be the correct result for a selective "\n            "swing system. Use the diagnostics below to see which entry conditions "\n            "blocked candidates before changing thresholds."\n        )\n''',
        "zero trade message",
    )

    marker = '''    if result.rejection_reasons:\n        with st.expander("Why signals were rejected by risk"):\n'''
    diagnostics = '''    if getattr(result, "strategy_blockers", None):\n        st.markdown("**Why candidate swings did not become signals**")\n        st.caption(\n            f"{getattr(result, 'evaluations', 0):,} strategy evaluations. "\n            "Counts can overlap because more than one required condition can fail "\n            "on the same bar."\n        )\n        st.dataframe(\n            pd.DataFrame(\n                sorted(result.strategy_blockers.items(), key=lambda item: -item[1])[:15],\n                columns=["Blocked by", "Evaluations"],\n            ),\n            width="stretch",\n            hide_index=True,\n        )\n\n'''
    text = replace_once(text, marker, diagnostics + marker, "dashboard blocker diagnostics")

    path.write_text(text)


if __name__ == "__main__":
    patch_engine()
    patch_dashboard()
    print("Applied swing backtest diagnostics/UI patch")

# Trigger marker: workflow installed after the script's initial commit.
