"""Logging configuration.

Three sinks, because they serve different readers:

* **Console** — concise, for the operator watching the bot run.
* ``logs/trading_bot.log`` — rotating full history for post-mortems.
* ``logs/events.jsonl`` — one JSON object per record, for programmatic analysis
  of why the bot did what it did.

:func:`log_signal_block` renders the multi-line decision block that makes the
human log readable at a glance.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-32s | %(funcName)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

#: Attributes present on every LogRecord; anything else is treated as custom
#: context and copied into the JSON payload.
_STANDARD_RECORD_FIELDS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
        "levelname", "levelno", "lineno", "message", "module", "msecs", "msg", "name",
        "pathname", "process", "processName", "relativeCreated", "stack_info",
        "taskName", "thread", "threadName",
    }
)

_configured = False


class JsonLinesFormatter(logging.Formatter):
    """Serialise a record as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_FIELDS and not key.startswith("_"):
                payload[key] = _json_safe(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def configure_logging(
    *,
    level: str = "INFO",
    log_dir: Path | str = "logs",
    json_enabled: bool = True,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    force: bool = False,
) -> logging.Logger:
    """Install console + file handlers on the root logger.

    Idempotent: calling it twice will not duplicate handlers unless ``force``.
    """
    global _configured
    root = logging.getLogger()
    if _configured and not force:
        return root
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    root.setLevel(logging.DEBUG)  # handlers apply their own thresholds

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(console)

    main_file = logging.handlers.RotatingFileHandler(
        directory / "trading_bot.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    main_file.setLevel(logging.DEBUG)
    main_file.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(main_file)

    error_file = logging.handlers.RotatingFileHandler(
        directory / "errors.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    error_file.setLevel(logging.ERROR)
    error_file.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(error_file)

    if json_enabled:
        json_file = logging.handlers.RotatingFileHandler(
            directory / "events.jsonl",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        json_file.setLevel(logging.DEBUG)
        json_file.setFormatter(JsonLinesFormatter())
        root.addHandler(json_file)

    # Third-party libraries are chatty at DEBUG; keep them at WARNING.
    for noisy in ("urllib3", "alpaca", "httpx", "websockets", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Module-level logger accessor."""
    return logging.getLogger(name)


def log_banner(logger: logging.Logger, title: str, fields: dict[str, Any]) -> None:
    """Log a boxed key/value block — used for startup and shutdown summaries."""
    width = 72
    lines = ["", "=" * width, f" {title}", "=" * width]
    for key, value in fields.items():
        lines.append(f" {key:<28}: {value}")
    lines.append("=" * width)
    logger.info("\n".join(lines))


def log_signal_block(
    logger: logging.Logger,
    *,
    symbol: str,
    strategy: str,
    direction: str,
    confidence: float,
    entry: float | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    risk_validation: str | None = None,
    position_size: int | None = None,
    reasons: list[str] | None = None,
    timestamp: datetime | None = None,
) -> None:
    """Render the human-readable signal block.

    Structured fields are attached to the record too, so ``events.jsonl`` stays
    machine-parseable while the text log stays readable.
    """
    stamp = (timestamp or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "",
        f"[{stamp}] {symbol} SIGNAL DETECTED",
        "",
        f"  Strategy        : {strategy}",
        f"  Direction       : {direction}",
        f"  Confidence      : {confidence:.0f}/100",
    ]
    if entry is not None:
        lines.append(f"  Entry           : ${entry:,.2f}")
    if stop_loss is not None:
        lines.append(f"  Stop Loss       : ${stop_loss:,.2f}")
    if take_profit is not None:
        lines.append(f"  Take Profit     : ${take_profit:,.2f}")
    if entry and stop_loss and take_profit:
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        if risk > 0:
            lines.append(f"  Risk/Reward     : 1:{reward / risk:.2f}")
    if reasons:
        lines.append("")
        lines.extend(f"  ✓ {reason}" for reason in reasons)
    if risk_validation:
        lines.append("")
        lines.append(f"  Risk Validation : {risk_validation}")
    if position_size is not None:
        lines.append(f"  Position Size   : {position_size} shares")
    lines.append("")

    logger.info(
        "\n".join(lines),
        extra={
            "event": "signal",
            "symbol": symbol,
            "strategy": strategy,
            "direction": direction,
            "confidence": confidence,
            "entry": entry,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size": position_size,
            "reasons": reasons or [],
        },
    )
