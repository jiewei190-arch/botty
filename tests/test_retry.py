"""Retry policy: what gets retried, what fails fast, and how backoff grows."""

from __future__ import annotations

import pytest
from alpaca.common.exceptions import APIError

from trading_bot.utils.retry import (
    RetryExhaustedError,
    chunked,
    is_retryable,
    retry_call,
)


class _StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_transient_statuses_are_retryable(status):
    assert is_retryable(_StatusError(status))


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retryable(status):
    assert not is_retryable(_StatusError(status))


def test_network_errors_are_retryable():
    assert is_retryable(ConnectionError("reset by peer"))
    assert is_retryable(TimeoutError("timed out"))


def test_api_error_without_a_status_is_retried():
    """A malformed APIError is more likely transport noise than a client bug."""
    assert is_retryable(APIError("something went wrong"))


def test_succeeds_without_retrying():
    calls = {"n": 0}

    def action():
        calls["n"] += 1
        return "ok"

    assert retry_call(action, sleep=lambda _: None) == "ok"
    assert calls["n"] == 1


def test_retries_until_success():
    calls = {"n": 0}

    def action():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("flaky")
        return "recovered"

    assert retry_call(action, sleep=lambda _: None) == "recovered"
    assert calls["n"] == 3


def test_non_retryable_error_propagates_immediately():
    calls = {"n": 0}

    def action():
        calls["n"] += 1
        raise _StatusError(401)

    with pytest.raises(_StatusError):
        retry_call(action, sleep=lambda _: None)
    assert calls["n"] == 1


def test_exhaustion_raises_with_context():
    def action():
        raise ConnectionError("always down")

    with pytest.raises(RetryExhaustedError, match="after 3 attempts"):
        retry_call(action, max_attempts=3, sleep=lambda _: None, description="fetch bars")


def test_backoff_is_exponential_and_capped():
    delays: list[float] = []

    def action():
        raise ConnectionError("down")

    with pytest.raises(RetryExhaustedError):
        retry_call(
            action, max_attempts=5, base_delay=1.0, max_delay=4.0, sleep=delays.append
        )

    assert len(delays) == 4
    assert delays[0] < delays[1] < delays[2]   # growing
    assert all(delay <= 4.0 * 1.25 for delay in delays)  # capped, allowing jitter


def test_max_attempts_must_be_positive():
    with pytest.raises(ValueError):
        retry_call(lambda: None, max_attempts=0)


def test_chunked_splits_evenly_and_keeps_the_remainder():
    assert list(chunked(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]
    assert list(chunked([], 3)) == []


def test_chunked_rejects_invalid_size():
    with pytest.raises(ValueError):
        list(chunked([1, 2], 0))
