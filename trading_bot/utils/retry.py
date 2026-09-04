"""Retry policy for broker and market-data calls.

A running bot must survive transient API failures. Rate limits (HTTP 429) and
gateway errors are retried with exponential backoff plus jitter; client errors
such as 401/403/404 are raised immediately because retrying cannot help.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from alpaca.common.exceptions import APIError

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: HTTP statuses worth retrying: rate limit and transient server/gateway faults.
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})


class RetryExhaustedError(RuntimeError):
    """Raised when every retry attempt failed."""


def _safe_getattr(obj: object, name: str) -> Any:
    """``getattr`` that also swallows exceptions raised *inside* a property.

    ``alpaca.common.exceptions.APIError`` exposes ``code`` as a property that
    JSON-decodes the error body; a non-JSON body (an HTML gateway error page, say)
    makes it raise. The retry classifier must never be the thing that crashes.
    """
    try:
        return getattr(obj, name, None)
    except Exception:  # noqa: BLE001 - defensive by design
        return None


def _status_code_of(error: BaseException) -> int | None:
    """Best-effort extraction of an HTTP status code from an exception."""
    for attribute in ("status_code", "code"):
        value = _safe_getattr(error, attribute)
        if isinstance(value, int):
            return value
    response = _safe_getattr(error, "response")
    status = _safe_getattr(response, "status_code") if response is not None else None
    return status if isinstance(status, int) else None


def is_retryable(error: BaseException) -> bool:
    """True when retrying ``error`` has a realistic chance of succeeding."""
    if isinstance(error, APIError):
        status = _status_code_of(error)
        # An APIError without a parseable status is usually a transport hiccup.
        return status is None or status in RETRYABLE_STATUS_CODES
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return True
    status = _status_code_of(error)
    return status in RETRYABLE_STATUS_CODES if status is not None else False


def retry_call(
    func: Callable[[], T],
    *,
    max_attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: Callable[[BaseException], bool] = is_retryable,
    description: str = "API call",
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``func`` with exponential backoff.

    Delays follow ``base_delay * 2**attempt`` capped at ``max_delay``, with up to
    25% random jitter so parallel symbol requests do not retry in lockstep.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    last_error: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as error:  # noqa: BLE001 - re-raised below
            last_error = error
            if not retry_on(error):
                logger.error("%s failed with a non-retryable error: %s", description, error)
                raise
            if attempt == max_attempts - 1:
                break
            delay = min(base_delay * (2**attempt), max_delay)
            delay *= 1 + random.random() * 0.25
            logger.warning(
                "%s failed (attempt %d/%d): %s. Retrying in %.1fs",
                description,
                attempt + 1,
                max_attempts,
                error,
                delay,
            )
            sleep(delay)

    raise RetryExhaustedError(
        f"{description} failed after {max_attempts} attempts: {last_error}"
    ) from last_error


def with_retry(
    *,
    max_attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: Callable[[BaseException], bool] = is_retryable,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator form of :func:`retry_call`."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return retry_call(
                lambda: func(*args, **kwargs),
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                retry_on=retry_on,
                description=func.__qualname__,
            )

        return wrapper

    return decorator


def chunked(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    """Split ``items`` into lists of at most ``size`` elements."""
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
