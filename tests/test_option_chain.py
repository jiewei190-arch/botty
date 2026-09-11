"""Tests for the Alpaca option-chain adapter.

This module had no tests, which is how a live-only defect shipped: a request
naming every contract in a chain is rejected by Alpaca with
``{"message":"symbol limit is 100"}``, and nothing exercised the adapter with a
chain big enough to trip it. The fakes here stand in for the SDK clients and
enforce the limit the way the API does, so the same mistake fails locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pytest

from trading_bot.config.settings import Settings
from trading_bot.options.alpaca import SYMBOL_REQUEST_LIMIT, AlpacaOptionChain, _batched


@dataclass
class FakeQuote:
    bid_price: float
    ask_price: float


@dataclass
class FakeGreeks:
    delta: float | None


class FakeSnapshot:
    def __init__(self, bid: float, ask: float, delta: float | None = 0.55) -> None:
        self.latest_quote = FakeQuote(bid_price=bid, ask_price=ask)
        self.greeks = FakeGreeks(delta=delta) if delta is not None else None
        self.implied_volatility = 0.32


@dataclass
class FakeContract:
    symbol: str
    type: str
    expiration_date: date
    strike_price: float
    open_interest: int = 500


class FakeContractResponse:
    def __init__(self, contracts: list[FakeContract]) -> None:
        self.option_contracts = contracts


class FakeTradingClient:
    def __init__(self, contracts: list[FakeContract]) -> None:
        self._contracts = contracts
        self.requests: list[object] = []

    def get_option_contracts(self, request):
        self.requests.append(request)
        return FakeContractResponse(self._contracts)


class FakeDataClient:
    """Rejects oversized requests exactly as Alpaca's endpoint does."""

    def __init__(self, snapshots: dict[str, FakeSnapshot]) -> None:
        self._snapshots = snapshots
        self.batch_sizes: list[int] = []

    def _check(self, symbols) -> list[str]:
        symbols = list(symbols)
        self.batch_sizes.append(len(symbols))
        if len(symbols) > SYMBOL_REQUEST_LIMIT:
            raise RuntimeError('{"message":"symbol limit is 100"}')
        return symbols

    def get_option_snapshot(self, request):
        symbols = self._check(request.symbol_or_symbols)
        return {s: self._snapshots[s] for s in symbols if s in self._snapshots}

    def get_option_latest_quote(self, request):
        symbols = self._check(request.symbol_or_symbols)
        return {
            s: self._snapshots[s].latest_quote
            for s in symbols
            if s in self._snapshots
        }


def build_chain(count: int, *, delta: float | None = 0.55):
    """A chain of ``count`` contracts and a matching adapter."""
    expiry = date.today() + timedelta(days=60)
    contracts = [
        FakeContract(
            symbol=f"NET{i:04d}C", type="call", expiration_date=expiry,
            strike_price=100.0 + i,
        )
        for i in range(count)
    ]
    snapshots = {c.symbol: FakeSnapshot(1.00, 1.10, delta) for c in contracts}
    data_client = FakeDataClient(snapshots)
    chain = AlpacaOptionChain(
        Settings(),
        trading_client=FakeTradingClient(contracts),
        data_client=data_client,
    )
    return chain, data_client, contracts


class TestBatching:
    def test_splits_at_the_limit(self):
        assert _batched(["A"] * 250) == [["A"] * 100, ["A"] * 100, ["A"] * 50]

    def test_a_short_list_is_one_batch(self):
        assert _batched(["A", "B"]) == [["A", "B"]]

    def test_an_empty_list_makes_no_request(self):
        assert _batched([]) == []

    def test_an_exact_multiple_makes_no_empty_trailing_batch(self):
        assert [len(b) for b in _batched(["A"] * 200)] == [100, 100]

    def test_order_is_preserved(self):
        symbols = [str(i) for i in range(250)]
        assert [s for batch in _batched(symbols) for s in batch] == symbols


class TestOversizedChain:
    def test_a_chain_over_the_limit_is_fetched_in_batches(self):
        """The NET/CI failure: 250 contracts in one request is rejected."""
        chain, data_client, _ = build_chain(250)

        quotes = chain.quotes("NET", "LONG", underlying_price=150.0)

        assert len(quotes) == 250
        assert data_client.batch_sizes == [100, 100, 50]
        assert max(data_client.batch_sizes) <= SYMBOL_REQUEST_LIMIT

    def test_every_contract_survives_the_round_trip(self):
        chain, _, contracts = build_chain(250)

        quotes = chain.quotes("NET", "LONG", underlying_price=150.0)

        assert [q.symbol for q in quotes] == [c.symbol for c in contracts]

    def test_a_chain_at_exactly_the_limit_still_works(self):
        chain, data_client, _ = build_chain(SYMBOL_REQUEST_LIMIT)

        quotes = chain.quotes("NET", "LONG", underlying_price=150.0)

        assert len(quotes) == SYMBOL_REQUEST_LIMIT
        assert data_client.batch_sizes == [SYMBOL_REQUEST_LIMIT]

    def test_a_small_chain_makes_exactly_one_request(self):
        chain, data_client, _ = build_chain(12)

        chain.quotes("NET", "LONG", underlying_price=150.0)

        assert data_client.batch_sizes == [12]

    def test_an_empty_chain_makes_no_snapshot_request(self):
        chain = AlpacaOptionChain(
            Settings(),
            trading_client=FakeTradingClient([]),
            data_client=FakeDataClient({}),
        )

        assert chain.quotes("NET", "LONG", underlying_price=150.0) == []
        assert chain.data_client.batch_sizes == []


class TestLatestMid:
    def test_supervising_many_contracts_is_also_batched(self):
        """The same defect, on the path that prices open positions."""
        chain, data_client, contracts = build_chain(180)

        result = chain.latest_mid([c.symbol for c in contracts])

        assert len(result) == 180
        assert data_client.batch_sizes == [100, 80]

    def test_no_symbols_makes_no_request(self):
        chain, data_client, _ = build_chain(5)

        assert chain.latest_mid([]) == {}
        assert data_client.batch_sizes == []

    def test_the_mid_is_the_midpoint(self):
        chain, _, contracts = build_chain(3)

        mid, bid, ask = chain.latest_mid([contracts[0].symbol])[contracts[0].symbol]

        assert bid == pytest.approx(1.00)
        assert ask == pytest.approx(1.10)
        assert mid == pytest.approx(1.05)


class TestSnapshotContents:
    def test_greeks_supply_the_delta_when_present(self):
        chain, _, _ = build_chain(3, delta=0.61)

        assert chain.quotes("NET", "LONG", underlying_price=150.0)[0].delta == pytest.approx(0.61)

    def test_a_missing_snapshot_drops_that_contract_only(self):
        chain, data_client, contracts = build_chain(120)
        del data_client._snapshots[contracts[0].symbol]

        quotes = chain.quotes("NET", "LONG", underlying_price=150.0)

        assert len(quotes) == 119
        assert contracts[0].symbol not in {q.symbol for q in quotes}

    def test_an_absent_greeks_block_leaves_delta_unset(self):
        """The indicative feed may omit greeks; selection falls back to moneyness."""
        chain, _, _ = build_chain(3, delta=None)

        assert chain.quotes("NET", "LONG", underlying_price=150.0)[0].delta is None
