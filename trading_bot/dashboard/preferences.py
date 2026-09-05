"""Small, local UI preferences that survive a restart.

Only one thing lives here so far, and it is the number every share count is
derived from: the balance of the account you trade.

Why persist it at all
---------------------
The balance is entered by hand, because the market-data provider and the broker
you trade need not be the same place — see ``RISK_ACCOUNT_EQUITY``. Entering it
per session is fine; retyping it from a hardcoded default every time the app
restarts is not, and a stale default silently sizes every position against a
balance you no longer have.

So the last value entered is remembered as the *starting point*, and stays
editable. It is a convenience, never a source of truth: the number in the box
is what sizing uses, whatever this file says.

Stored beside the database rather than in ``.env`` on purpose. ``.env`` is
configuration a person edits deliberately; this is a UI convenience the app
writes on its own, and the two should not be mixed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PREFERENCES_FILE = "dashboard_preferences.json"


def _path(directory: Path) -> Path:
    return Path(directory) / PREFERENCES_FILE


def load_preferences(directory: Path) -> dict[str, Any]:
    """Read stored preferences, returning ``{}` when there are none.

    Never raises: a preference file that cannot be read is a reason to fall
    back to defaults, not to stop the app from starting.
    """
    path = _path(directory)
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        logger.warning("Ignoring unreadable preferences at %s: %s", path, error)
        return {}
    return loaded if isinstance(loaded, dict) else {}


def save_preference(directory: Path, key: str, value: Any) -> None:
    """Update one preference, leaving the rest alone."""
    path = _path(directory)
    current = load_preferences(directory)
    if current.get(key) == value:
        return
    current[key] = value
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, indent=2, sort_keys=True))
    except OSError as error:
        logger.warning("Could not save preferences to %s: %s", path, error)


def remembered_equity(directory: Path, fallback: float) -> float:
    """The last equity entered, or ``fallback`` when there is none or it is bad."""
    value = load_preferences(directory).get("account_equity")
    try:
        equity = float(value)
    except (TypeError, ValueError):
        return fallback
    return equity if equity > 0 else fallback
