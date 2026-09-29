import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from robottraderslab import Symbol
from robottraderslab.backtester.simulator import (
    SimulatedFuturesExchange,
)
from robottraderslab.strategies import (
    AccountSnapshots,
    BookKeeper,
    Execution,
    OHLCVs,
    OrderSide,
    PositionSide,
    StopLoss,
    StrategyRequirements,
    TradingMode,
    TradingSystem,
)
from robottraderslab.strategies.futures import (
    FuturesAccount,
    FuturesMarketOrderAction,
    FuturesOrderBatchAction,
)
from robottraderslab_envelope import EnvelopeStrategy
from robottraderslab_envelope.reentry_guard import ReentryGuard

BTC = Symbol.create("BTC/USDT:USDT")
ETH = Symbol.create("ETH/USDT:USDT")
TIMEFRAME = "1h"
BAR_START = datetime(2024, 1, 1)
REFERENCE = 100.0
ENVELOPE = 0.05


def _make_profile(symbol: Symbol) -> dict[str, Any]:
    return {
        "symbol": str(symbol),
        "timeframe": TIMEFRAME,
        "average_type": "SMA",
        "average_period": 3,
        "envelopes": [ENVELOPE],
        "stop_loss_pct": 0.04,
        "total_balance_ratio": 0.3,
        "leverage": 5.0,
        "margin_mode": "isolated",
    }


def _build_ohlcvs(
    closes_by_symbol: dict[Symbol, list[float]],
) -> tuple[OHLCVs, datetime]:
    n_bars = len(next(iter(closes_by_symbol.values())))
    dates = pd.date_range(BAR_START, periods=n_bars, freq="1h")
    frames = {
        symbol: pd.DataFrame(
            {
                "open": closes,
                "high": [c + 0.5 for c in closes],
                "low": [c - 0.5 for c in closes],
                "close": closes,
                "volume": [1000.0] * n_bars,
            },
            index=dates,
        )
        for symbol, closes in closes_by_symbol.items()
    }
    ohlcvs = OHLCVs({TIMEFRAME: frames})
    for symbol in closes_by_symbol:
        profile_id = str(symbol)
        ohlcvs.add_column(
            symbol, TIMEFRAME, f"{profile_id}_reference", [REFERENCE] * n_bars
        )
        ohlcvs.add_column(
            symbol,
            TIMEFRAME,
            f"{profile_id}_band_low_1",
            [REFERENCE * (1 - ENVELOPE)] * n_bars,
        )
        ohlcvs.add_column(
            symbol,
            TIMEFRAME,
            f"{profile_id}_band_high_1",
            [REFERENCE * (1 + ENVELOPE)] * n_bars,
        )
    last_timestamp = BAR_START
    for snapshot in ohlcvs._iter_timeframes():
        last_timestamp = snapshot.timestamp
    return ohlcvs, last_timestamp


def _entry_rungs(bookkeeper: BookKeeper, symbol: Symbol, side: OrderSide) -> list:
    return [
        order
        for action in bookkeeper.list_actions()
        if isinstance(action, FuturesOrderBatchAction)
        for order in action.orders
        if isinstance(order, FuturesMarketOrderAction)
        and order.symbol == symbol
        and order.side == side
        and not order.reduce_only
    ]


class _StrategyHarness:
    def __init__(self, profiles: list[dict[str, Any]]) -> None:
        self.exchange = SimulatedFuturesExchange.create_from_settings(
            initial_balance={"USDT": 10_000.0},
            maker_fee_rate=0.0,
            taker_fee_rate=0.0,
            fee_mode="cost",
        )
        self.engine = self.exchange.simulation_engine
        self.account = FuturesAccount(exchange=self.exchange)
        self.requirements = StrategyRequirements()
        self.strategy = EnvelopeStrategy(
            account=self.account,
            trading_system=TradingSystem(trading_mode=TradingMode.BACKTEST),
            config_dir=Path("bot-config-dir"),
            profiles=profiles,
        )

    def setup(self) -> None:
        asyncio.run(self.strategy.setup(self.requirements))

    def account_state(self, since: datetime) -> AccountSnapshots:
        requirement = self.requirements.account._get_all()[0]
        account_snapshot = asyncio.run(
            self.account._snapshot(requirement, executions_since=since)
        )
        return AccountSnapshots({self.account.name: account_snapshot})

    def record_sl_fill(
        self, symbol: Symbol, side: PositionSide, timestamp: datetime
    ) -> None:
        closing_side = OrderSide.SELL if side == PositionSide.LONG else OrderSide.BUY
        self.engine._executions.append(
            Execution(
                execution_id="sl-1",
                order_id="sl-1",
                symbol=symbol,
                side=closing_side,
                price=100.0,
                quantity=1.0,
                timestamp=timestamp,
                kind="stop-loss",
            )
        )


@pytest.fixture
def btc_harness() -> _StrategyHarness:
    harness = _StrategyHarness([_make_profile(BTC)])
    harness.setup()
    return harness


class TestReentryLifecycleIntegration:
    def test_blocks_long_rungs_after_sl_fills(self, btc_harness):
        btc_harness.record_sl_fill(
            BTC, PositionSide.LONG, BAR_START + timedelta(hours=4)
        )
        ohlcvs, last_timestamp = _build_ohlcvs(
            {BTC: [100.0] * 5 + [95.0] * 5}  # close stays below MA after fire
        )
        bookkeeper = BookKeeper()

        btc_harness.strategy.book_trading_actions(
            ohlcvs,
            btc_harness.account_state(BAR_START),
            last_timestamp,
            bookkeeper,
            [TIMEFRAME],
        )

        assert _entry_rungs(bookkeeper, BTC, OrderSide.BUY) == []
        assert len(_entry_rungs(bookkeeper, BTC, OrderSide.SELL)) == 1

    def test_releases_when_close_crosses_back_above_ma(self, btc_harness):
        btc_harness.record_sl_fill(
            BTC, PositionSide.LONG, BAR_START + timedelta(hours=4)
        )
        ohlcvs, last_timestamp = _build_ohlcvs(
            {BTC: [100.0] * 5 + [95.0, 96.0, 97.0, 101.0, 100.0]}
        )
        bookkeeper = BookKeeper()

        btc_harness.strategy.book_trading_actions(
            ohlcvs,
            btc_harness.account_state(BAR_START),
            last_timestamp,
            bookkeeper,
            [TIMEFRAME],
        )

        assert len(_entry_rungs(bookkeeper, BTC, OrderSide.BUY)) == 1
        assert len(_entry_rungs(bookkeeper, BTC, OrderSide.SELL)) == 1


class TestReentrySymbolIsolation:
    def test_sl_on_btc_does_not_gate_eth(self):
        harness = _StrategyHarness([_make_profile(BTC), _make_profile(ETH)])
        harness.setup()
        harness.record_sl_fill(BTC, PositionSide.LONG, BAR_START + timedelta(hours=4))
        ohlcvs, last_timestamp = _build_ohlcvs(
            {
                BTC: [100.0] * 5 + [95.0] * 5,
                ETH: [100.0] * 5 + [95.0] * 5,
            }
        )
        bookkeeper = BookKeeper()

        harness.strategy.book_trading_actions(
            ohlcvs,
            harness.account_state(BAR_START),
            last_timestamp,
            bookkeeper,
            [TIMEFRAME],
        )

        assert _entry_rungs(bookkeeper, BTC, OrderSide.BUY) == []
        assert len(_entry_rungs(bookkeeper, ETH, OrderSide.BUY)) == 1


def _candles_around_a_stop(closes_from_19h: list[float]) -> OHLCVs:
    """1h candles from 17:00 to 22:00, a long stop at 95 firing on the candle
    opened at 19:00.
    """
    closes = [100.0, 100.0, *closes_from_19h]
    frame = pd.DataFrame(
        {
            "open": closes,
            "high": [close + 0.5 for close in closes],
            "low": [99.5, 99.5, 90.0, 98.5, 98.5],
            "close": closes,
            "volume": [1000.0] * 5,
        },
        index=pd.date_range("2024-01-01 17:00", periods=5, freq="1h"),
    )
    ohlcvs = OHLCVs({TIMEFRAME: {BTC: frame}})
    ohlcvs.add_column(BTC, TIMEFRAME, "reference", [REFERENCE] * 5)
    return ohlcvs


def _simulated_stop_loss(ohlcvs: OHLCVs) -> Execution:
    exchange = SimulatedFuturesExchange.create_from_settings(
        initial_balance={"USDT": 10_000.0},
        maker_fee_rate=0.0,
        taker_fee_rate=0.0,
        fee_mode="cost",
    )
    engine = exchange.simulation_engine
    engine.enable_execution_recording()
    asyncio.run(
        exchange.place_market_order(
            BTC, OrderSide.BUY, 1.0, stop_loss=StopLoss(trigger_price=95.0)
        )
    )
    for snapshot in ohlcvs._iter_timeframes():
        engine.simulate_on_current_ohlcvs(snapshot.timestamp, snapshot.ohlcvs_by_symbol)
    return next(
        execution
        for execution in engine.get_executions_since(BAR_START)
        if execution.kind == "stop-loss"
    )


def _live_stop_loss() -> Execution:
    return Execution(
        execution_id="sl-live",
        order_id="sl-live",
        symbol=BTC,
        side=OrderSide.SELL,
        price=95.0,
        quantity=1.0,
        timestamp=datetime(2024, 1, 1, 19, 23),
        kind="stop-loss",
    )


def _blocked_sides(ohlcvs: OHLCVs, fill: Execution) -> frozenset[PositionSide]:
    state = ReentryGuard().state_for(
        BTC, TIMEFRAME, "reference", ohlcvs, [fill], datetime(2024, 1, 1, 22)
    )
    return state.blocked_sides


class TestReentryLiveAndBacktestAgree:
    def test_the_simulated_stop_carries_the_close_of_its_candle(self):
        ohlcvs = _candles_around_a_stop([101.0, 99.0, 99.0])

        stop_loss = _simulated_stop_loss(ohlcvs)

        assert stop_loss.timestamp == datetime(2024, 1, 1, 20)

    def test_the_candle_the_stop_fired_in_does_not_clear_it(self):
        ohlcvs = _candles_around_a_stop([101.0, 99.0, 99.0])

        simulated = _blocked_sides(ohlcvs, _simulated_stop_loss(ohlcvs))
        live = _blocked_sides(ohlcvs, _live_stop_loss())

        assert simulated == frozenset({PositionSide.LONG})
        assert live == frozenset({PositionSide.LONG})

    def test_the_first_candle_after_the_stop_clears_it(self):
        ohlcvs = _candles_around_a_stop([99.0, 101.0, 99.0])

        simulated = _blocked_sides(ohlcvs, _simulated_stop_loss(ohlcvs))
        live = _blocked_sides(ohlcvs, _live_stop_loss())

        assert simulated == frozenset()
        assert live == frozenset()
