import asyncio
from datetime import datetime

import pandas as pd
from robottraderslab_envelope import EnvelopeStrategy

from robottraderslab import Symbol
from robottraderslab.strategies import (
    AccountSnapshot,
    AccountSnapshots,
    Balance,
    BookKeeper,
    OHLCVs,
    StrategyRequirements,
    TradingMode,
)
from robottraderslab.strategies.futures import (
    FuturesMarketOrderAction,
    FuturesOrderBatchAction,
    MarginMode,
    MarginSettings,
)

BTC = Symbol.create("BTC/USDT:USDT")
ETH = Symbol.create("ETH/USDT:USDT")
ACCOUNT_NAME = "test"
TIMEFRAME = "1d"
TIMESTAMP = datetime(2024, 2, 19)

BTC_PROFILE = {
    "symbol": "BTC/USDT:USDT",
    "timeframe": TIMEFRAME,
    "average_type": "SMA",
    "average_period": 100,
    "envelopes": [0.05],
    "stop_loss_pct": 0.04,
    "total_balance_ratio": 0.25,
    "leverage": 5.0,
    "margin_mode": "isolated",
}
ETH_PROFILE = {**BTC_PROFILE, "symbol": "ETH/USDT:USDT"}


def _stand_on_first_candle(ohlcvs: OHLCVs) -> None:
    """Set the iteration cursor so `current` and `signal` read the first candle."""
    next(ohlcvs._iter_timeframes())


def _make_ohlcvs(symbols_with_columns: list[Symbol]) -> OHLCVs:
    dates = pd.date_range("2024-02-19", periods=1, freq="1D")
    frames = {
        symbol: pd.DataFrame(
            {
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            },
            index=dates,
        )
        for symbol in (BTC, ETH)
    }
    ohlcvs = OHLCVs({TIMEFRAME: frames})

    for symbol in symbols_with_columns:
        profile_id = str(symbol)
        ohlcvs.add_column(symbol, TIMEFRAME, f"{profile_id}_reference", [100.0])
        ohlcvs.add_column(symbol, TIMEFRAME, f"{profile_id}_band_low_1", [95.0])
        ohlcvs.add_column(symbol, TIMEFRAME, f"{profile_id}_band_high_1", [105.0])

    _stand_on_first_candle(ohlcvs)

    return ohlcvs


def _account_state(*symbols: Symbol) -> AccountSnapshot:
    snapshot = AccountSnapshot(
        account_name=ACCOUNT_NAME,
        balances={"USDT": Balance(locked=0.0, total=10_000.0)},
        open_orders=[],
        positions={},
        executions=[],
        executions_declared=True,
        margin_settings={
            symbol: MarginSettings(leverage=None, margin_mode=MarginMode.CROSS)
            for symbol in symbols
        },
        conversion_rates={symbol: 1.0 for symbol in symbols},
    )
    return AccountSnapshots({ACCOUNT_NAME: snapshot})


def _rung_orders_for(bookkeeper: BookKeeper, symbol: Symbol) -> list:
    return [
        order
        for action in bookkeeper.list_actions()
        if isinstance(action, FuturesOrderBatchAction)
        for order in action.orders
        if isinstance(order, FuturesMarketOrderAction) and order.symbol == symbol
    ]


def _load(strategy: EnvelopeStrategy) -> EnvelopeStrategy:
    asyncio.run(strategy.setup(StrategyRequirements()))
    return strategy


class TestLiveModeIsolation:
    def test_profile_with_internal_error(self, make_strategy):
        strategy = _load(
            make_strategy([BTC_PROFILE, ETH_PROFILE], trading_mode=TradingMode.LIVE)
        )
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs([ETH]),
            _account_state(BTC, ETH),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        assert not _rung_orders_for(bookkeeper, BTC)
        assert _rung_orders_for(bookkeeper, ETH)


class TestSharedExchangeState:
    def test_single_snapshot_serves_all_profiles(self, make_strategy):
        strategy = _load(make_strategy([BTC_PROFILE, ETH_PROFILE]))
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs([BTC, ETH]),
            _account_state(BTC, ETH),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        assert _rung_orders_for(bookkeeper, BTC)
        assert _rung_orders_for(bookkeeper, ETH)
