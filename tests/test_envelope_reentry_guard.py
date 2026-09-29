import logging
from datetime import datetime, timedelta

import pandas as pd
import pytest

from robottraderslab import Symbol
from robottraderslab.strategies import Execution, OHLCVs, OrderSide, PositionSide
from robottraderslab_envelope.reentry_guard import ReentryGuard, ReentryState

BTC = Symbol.create("BTC/USDT:USDT")
ETH = Symbol.create("ETH/USDT:USDT")
TIMEFRAME = "1h"
REFERENCE_COLUMN = "reference"
BAR_START = "2024-01-01"


def _bars(count: int) -> pd.DatetimeIndex:
    return pd.date_range(BAR_START, periods=count, freq="1h")


def _moments(count: int) -> pd.DatetimeIndex:
    return _bars(count) + pd.Timedelta(hours=1)


def _ohlcvs(
    closes: list[float], references: list[float], symbol: Symbol = BTC
) -> OHLCVs:
    frame = pd.DataFrame({"close": closes}, index=_bars(len(closes)))
    ohlcvs = OHLCVs({TIMEFRAME: {symbol: frame}})
    ohlcvs.add_column(symbol, TIMEFRAME, REFERENCE_COLUMN, references)
    return ohlcvs


def _long_sl_fill(symbol: Symbol, timestamp: datetime) -> Execution:
    return Execution(
        execution_id="sl-long",
        order_id="sl-long",
        symbol=symbol,
        side=OrderSide.SELL,
        price=100.0,
        quantity=1.0,
        timestamp=timestamp,
        kind="stop-loss",
    )


def _short_sl_fill(symbol: Symbol, timestamp: datetime) -> Execution:
    return Execution(
        execution_id="sl-short",
        order_id="sl-short",
        symbol=symbol,
        side=OrderSide.BUY,
        price=100.0,
        quantity=1.0,
        timestamp=timestamp,
        kind="stop-loss",
    )


class TestReentryState:
    def test_allows_entry_when_side_not_locked(self):
        state = ReentryState(blocked_sides=frozenset())

        assert state.is_entry_allowed(PositionSide.LONG) is True
        assert state.is_entry_allowed(PositionSide.SHORT) is True

    def test_blocks_long_when_long_locked(self):
        state = ReentryState(blocked_sides=frozenset({PositionSide.LONG}))

        assert state.is_entry_allowed(PositionSide.LONG) is False
        assert state.is_entry_allowed(PositionSide.SHORT) is True

    def test_blocks_short_when_short_locked(self):
        state = ReentryState(blocked_sides=frozenset({PositionSide.SHORT}))

        assert state.is_entry_allowed(PositionSide.LONG) is True
        assert state.is_entry_allowed(PositionSide.SHORT) is False


class TestReentryGuard:
    @pytest.fixture
    def guard(self) -> ReentryGuard:
        return ReentryGuard()

    def test_returns_empty_when_no_fills(self, guard):
        moments = _moments(3)
        ohlcvs = _ohlcvs([100.0, 101.0, 102.0], [100.0, 100.0, 100.0])

        state = guard.state_for(
            BTC, TIMEFRAME, REFERENCE_COLUMN, ohlcvs, [], moments[-1]
        )

        assert state.blocked_sides == frozenset()

    def test_returns_empty_when_close_series_empty(self, guard):
        ohlcvs = _ohlcvs([], [])

        state = guard.state_for(
            BTC, TIMEFRAME, REFERENCE_COLUMN, ohlcvs, [], datetime(2024, 1, 1)
        )

        assert state.blocked_sides == frozenset()

    def test_locks_long_when_long_stop_loss_fills_and_no_clearance(self, guard):
        moments = _moments(3)
        ohlcvs = _ohlcvs([100.0, 99.0, 98.0], [100.0, 100.0, 100.0])
        fills = [_long_sl_fill(BTC, moments[0])]

        state = guard.state_for(
            BTC, TIMEFRAME, REFERENCE_COLUMN, ohlcvs, fills, moments[-1]
        )

        assert state.blocked_sides == frozenset({PositionSide.LONG})

    def test_locks_short_when_short_stop_loss_fills_and_no_clearance(self, guard):
        moments = _moments(3)
        ohlcvs = _ohlcvs([100.0, 101.0, 102.0], [100.0, 100.0, 100.0])
        fills = [_short_sl_fill(BTC, moments[0])]

        state = guard.state_for(
            BTC, TIMEFRAME, REFERENCE_COLUMN, ohlcvs, fills, moments[-1]
        )

        assert state.blocked_sides == frozenset({PositionSide.SHORT})

    def test_unblocks_long_when_close_rises_back_above_reference(self, guard):
        moments = _moments(3)
        ohlcvs = _ohlcvs([99.0, 98.0, 101.0], [100.0, 100.0, 100.0])
        fills = [_long_sl_fill(BTC, moments[0])]

        state = guard.state_for(
            BTC, TIMEFRAME, REFERENCE_COLUMN, ohlcvs, fills, moments[-1]
        )

        assert state.blocked_sides == frozenset()

    def test_unblocks_short_when_close_falls_back_below_reference(self, guard):
        moments = _moments(3)
        ohlcvs = _ohlcvs([101.0, 102.0, 99.0], [100.0, 100.0, 100.0])
        fills = [_short_sl_fill(BTC, moments[0])]

        state = guard.state_for(
            BTC, TIMEFRAME, REFERENCE_COLUMN, ohlcvs, fills, moments[-1]
        )

        assert state.blocked_sides == frozenset()

    def test_ignores_bars_after_the_current_one(self, guard):
        moments = _moments(3)
        ohlcvs = _ohlcvs([99.0, 98.0, 101.0], [100.0, 100.0, 100.0])
        fills = [_long_sl_fill(BTC, moments[0])]

        state = guard.state_for(
            BTC, TIMEFRAME, REFERENCE_COLUMN, ohlcvs, fills, moments[1]
        )

        assert state.blocked_sides == frozenset({PositionSide.LONG})

    def test_ignores_the_bar_the_stop_loss_fired_on(self, guard):
        moments = _moments(2)
        ohlcvs = _ohlcvs([101.0, 99.0], [100.0, 100.0])
        fills = [_long_sl_fill(BTC, moments[0])]

        state = guard.state_for(
            BTC, TIMEFRAME, REFERENCE_COLUMN, ohlcvs, fills, moments[-1]
        )

        assert state.blocked_sides == frozenset({PositionSide.LONG})

    def test_uses_most_recent_fill_per_side(self, guard):
        moments = _moments(5)
        ohlcvs = _ohlcvs(
            [100.0, 101.0, 102.0, 99.0, 98.0], [100.0, 100.0, 100.0, 100.0, 100.0]
        )
        fills = [_long_sl_fill(BTC, moments[0]), _long_sl_fill(BTC, moments[3])]

        state = guard.state_for(
            BTC, TIMEFRAME, REFERENCE_COLUMN, ohlcvs, fills, moments[-1]
        )

        assert state.blocked_sides == frozenset({PositionSide.LONG})

    def test_locks_both_sides_independently(self, guard):
        moments = _moments(4)
        ohlcvs = _ohlcvs([100.0, 99.0, 100.5, 101.0], [100.0, 100.0, 100.0, 100.0])
        fills = [_long_sl_fill(BTC, moments[0]), _short_sl_fill(BTC, moments[2])]

        state = guard.state_for(
            BTC, TIMEFRAME, REFERENCE_COLUMN, ohlcvs, fills, moments[-1]
        )

        assert state.blocked_sides == frozenset({PositionSide.SHORT})

    def test_ignores_fills_for_other_symbols(self, guard):
        moments = _moments(3)
        ohlcvs = _ohlcvs([100.0, 99.0, 98.0], [100.0, 100.0, 100.0])
        fills = [_long_sl_fill(ETH, moments[0])]

        state = guard.state_for(
            BTC, TIMEFRAME, REFERENCE_COLUMN, ohlcvs, fills, moments[-1]
        )

        assert state.blocked_sides == frozenset()

    def test_ignores_fills_before_loaded_window(self, guard):
        moments = _moments(3)
        ohlcvs = _ohlcvs([100.0, 99.0, 98.0], [100.0, 100.0, 100.0])
        fills = [_long_sl_fill(BTC, _bars(3)[0] - timedelta(minutes=1))]

        state = guard.state_for(
            BTC, TIMEFRAME, REFERENCE_COLUMN, ohlcvs, fills, moments[-1]
        )

        assert state.blocked_sides == frozenset()

    def test_reads_the_candles_of_the_latest_store(self, guard):
        moments = _moments(3)
        stale = _ohlcvs([100.0, 99.0, 98.0], [100.0, 100.0, 100.0])
        fresh = _ohlcvs([100.0, 99.0, 101.0], [100.0, 100.0, 100.0])
        fills = [_long_sl_fill(BTC, moments[0])]

        guard.state_for(BTC, TIMEFRAME, REFERENCE_COLUMN, stale, fills, moments[-1])
        state = guard.state_for(
            BTC, TIMEFRAME, REFERENCE_COLUMN, fresh, fills, moments[-1]
        )

        assert state.blocked_sides == frozenset()

    def test_logs_suspension_once_while_block_persists(self, guard, caplog):
        moments = _moments(3)
        ohlcvs = _ohlcvs([100.0, 99.0, 98.0], [100.0, 100.0, 100.0])
        fills = [_long_sl_fill(BTC, moments[0])]

        with caplog.at_level(logging.INFO):
            guard.state_for(BTC, TIMEFRAME, REFERENCE_COLUMN, ohlcvs, fills, moments[1])
            guard.state_for(BTC, TIMEFRAME, REFERENCE_COLUMN, ohlcvs, fills, moments[2])

        assert len(caplog.records) == 1
        assert "long entries suspended" in caplog.records[0].message

    def test_logs_resumption_when_block_clears(self, guard, caplog):
        moments = _moments(3)
        ohlcvs = _ohlcvs([99.0, 98.0, 101.0], [100.0, 100.0, 100.0])
        fills = [_long_sl_fill(BTC, moments[0])]

        with caplog.at_level(logging.INFO):
            guard.state_for(BTC, TIMEFRAME, REFERENCE_COLUMN, ohlcvs, fills, moments[1])
            guard.state_for(BTC, TIMEFRAME, REFERENCE_COLUMN, ohlcvs, fills, moments[2])

        messages = [record.message for record in caplog.records]
        assert len(messages) == 2
        assert "long entries suspended" in messages[0]
        assert "long entries resumed" in messages[1]

    def test_no_logging_while_never_locked(self, guard, caplog):
        moments = _moments(3)
        ohlcvs = _ohlcvs([100.0, 101.0, 102.0], [100.0, 100.0, 100.0])

        with caplog.at_level(logging.INFO):
            guard.state_for(BTC, TIMEFRAME, REFERENCE_COLUMN, ohlcvs, [], moments[-1])

        assert len(caplog.records) == 0
