import asyncio
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd

from robottraderslab import Symbol, TimeFrame
from robottraderslab.backtester.backtester import Backtester
from robottraderslab.backtester.simulator import (
    CacheFillRecorder,
    SimulatedFuturesExchange,
)
from robottraderslab.exchanges import OHLCVProviderProtocol, PositionSnapshot
from robottraderslab.live.livebot import LiveBot
from robottraderslab.strategies import (
    Execution,
    OrderSide,
    TradingMode,
    TradingSystem,
)
from robottraderslab.strategies.futures import FuturesAccount
from robottraderslab_envelope import EnvelopeStrategy

BTC = Symbol.create("BTC/USDT:USDT")
TIMEFRAME: TimeFrame = "1m"
FIRST_OPEN = datetime(2024, 1, 1, tzinfo=UTC)
QUIET_HIGHS = [100.5] * 10
SHORT_FILLED_HIGH = 106.0
SHORT_STOPPED_HIGH = 110.0
HIGHS = [*QUIET_HIGHS, SHORT_FILLED_HIGH, SHORT_STOPPED_HIGH]
LAST_CLOSE = FIRST_OPEN + timedelta(minutes=len(HIGHS))
STOPPED_AFTER_CLOSE = LAST_CLOSE + timedelta(seconds=1)
SHORT_STOP = 109.47
PROFILE: dict[str, Any] = {
    "symbol": str(BTC),
    "timeframe": TIMEFRAME,
    "average_type": "SMA",
    "average_period": 3,
    "envelopes": [0.05],
    "stop_loss_pct": 0.04,
    "total_balance_ratio": 0.3,
    "leverage": 5.0,
    "margin_mode": "isolated",
}

type Rung = tuple[OrderSide, float]


def _candles() -> pd.DataFrame:
    candle_count = len(HIGHS)
    return pd.DataFrame(
        {
            "open": [100.0] * candle_count,
            "high": HIGHS,
            "low": [99.5] * candle_count,
            "close": [100.0] * candle_count,
            "volume": [1000.0] * candle_count,
        },
        index=pd.date_range(FIRST_OPEN, periods=candle_count, freq="1min"),
    )


def _provider() -> Mock:
    provider = Mock(spec=OHLCVProviderProtocol)
    provider.fetch_ohlcv.return_value = _candles()
    return provider


def _exchange[E: SimulatedFuturesExchange](kind: type[E]) -> E:
    return kind.create_from_settings(
        initial_balance={"USDT": 10_000.0},
        maker_fee_rate=0.0,
        taker_fee_rate=0.0,
        fee_mode="cost",
    )


def _strategy(exchange: SimulatedFuturesExchange) -> EnvelopeStrategy:
    return EnvelopeStrategy(
        account=FuturesAccount(exchange=exchange),
        trading_system=TradingSystem(trading_mode=TradingMode.BACKTEST),
        config_dir=Path("bot-config-dir"),
        profiles=[PROFILE],
        entry_order="limit",
    )


def _resting_rungs(exchange: SimulatedFuturesExchange) -> list[Rung]:
    resting = asyncio.run(exchange.get_open_orders([BTC]))
    return sorted(
        (order.side, order.limit_price) for order in resting if not order.reduce_only
    )


def _stop_loss_fill(timestamp: datetime) -> Execution:
    return Execution(
        execution_id="sl-short",
        order_id="sl-short",
        symbol=BTC,
        side=OrderSide.BUY,
        price=SHORT_STOP,
        quantity=1.0,
        timestamp=timestamp,
        kind="stop-loss",
    )


def _book_in_the_backtest() -> list[Rung]:
    exchange = _exchange(SimulatedFuturesExchange)
    backtester = Backtester(
        _strategy(exchange),
        exchange.simulation_engine,
        _provider(),
        CacheFillRecorder(),
        FIRST_OPEN.date().isoformat(),
        LAST_CLOSE.date().isoformat(),
    )
    asyncio.run(backtester.run())
    return _resting_rungs(exchange)


class _ExchangeFillingAStop(SimulatedFuturesExchange):
    stop: Execution


class _ExchangeReportingALateStop(_ExchangeFillingAStop):
    """A venue whose executions read holds a stop-loss it filled after the
    cycle's last candle closed.
    """

    async def get_executions_since(
        self,
        since: datetime,
        symbols: Iterable[Symbol],
    ) -> list[Execution]:
        executions = await super().get_executions_since(since, symbols)
        return [*executions, self.stop]


class _ExchangeFillingAStopWhilePositionsAnswer(_ExchangeFillingAStop):
    """A venue whose stop-loss fills while it answers the positions read, so
    only an executions read made after that answer holds it.
    """

    positions_answered: bool = False

    async def get_open_positions(
        self, symbols: Iterable[Symbol]
    ) -> dict[Symbol, PositionSnapshot]:
        positions = await super().get_open_positions(symbols)
        await asyncio.sleep(0)
        self.positions_answered = True
        return positions

    async def get_executions_since(
        self,
        since: datetime,
        symbols: Iterable[Symbol],
    ) -> list[Execution]:
        executions = await super().get_executions_since(since, symbols)
        if not self.positions_answered:
            return executions
        return [*executions, self.stop]


def _book_in_a_live_cycle(
    stop_loss_filled_at: datetime,
    kind: type[_ExchangeFillingAStop] = _ExchangeReportingALateStop,
) -> list[Rung]:
    exchange = _exchange(kind)
    exchange.stop = _stop_loss_fill(stop_loss_filled_at)
    exchange.simulation_engine.enable_execution_recording()
    asyncio.run(LiveBot(_strategy(exchange), _provider()).run())
    return _resting_rungs(exchange)


class TestStopLossFilledAfterTheClose:
    def test_the_stopped_side_gets_no_rung_in_a_live_cycle(self):
        rungs = _book_in_a_live_cycle(STOPPED_AFTER_CLOSE)

        assert [side for side, _ in rungs] == [OrderSide.BUY]

    def test_a_live_cycle_books_the_rungs_a_backtest_books(self):
        backtest = _book_in_the_backtest()

        live = _book_in_a_live_cycle(STOPPED_AFTER_CLOSE)

        assert live == backtest
        assert [side for side, _ in backtest] == [OrderSide.BUY]


class TestStopLossFilledBetweenTheTwoAnswers:
    def test_the_stopped_side_gets_no_rung_in_a_live_cycle(self):
        rungs = _book_in_a_live_cycle(
            STOPPED_AFTER_CLOSE, _ExchangeFillingAStopWhilePositionsAnswer
        )

        assert [side for side, _ in rungs] == [OrderSide.BUY]

    def test_a_live_cycle_books_the_rungs_a_backtest_books(self):
        backtest = _book_in_the_backtest()

        live = _book_in_a_live_cycle(
            STOPPED_AFTER_CLOSE, _ExchangeFillingAStopWhilePositionsAnswer
        )

        assert live == backtest
