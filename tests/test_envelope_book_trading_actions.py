import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest

from robottraderslab import Symbol
from robottraderslab.exchanges import OrderType
from robottraderslab.strategies import (
    AccountSnapshot,
    AccountSnapshots,
    Balance,
    BookKeeper,
    Execution,
    OHLCVs,
    OrderProtocol,
    OrderSide,
    PositionSide,
    StrategyRequirements,
)
from robottraderslab.strategies.futures import (
    BatchableOrderAction,
    CancelOrderByIdAction,
    CancelOrdersForSymbolAction,
    FuturesLimitOrderAction,
    FuturesMarketOrderAction,
    FuturesOrderBatchAction,
    FuturesOrderModifyAction,
    MarginMode,
    MarginSettings,
    OrderModification,
    PositionSnapshot,
    SetLeverageAction,
    SetMarginModeAction,
    UpdatePositionStopLossAction,
    UpdatePositionTakeProfitAction,
)
from robottraderslab_envelope import EnvelopeStrategy

BTC = Symbol.create("BTC/USDT:USDT")
ACCOUNT_NAME = "test"
TIMEFRAME = "1d"
TIMESTAMP = datetime(2024, 2, 19)
SETUP_ID = "BTC/USDT:USDT"
_FAKE_PROCESS_TAG = "fakeprocess99-1"


def _stand_on_first_candle(ohlcvs: OHLCVs) -> None:
    """Set the iteration cursor so `current` and `signal` read the first candle."""
    next(ohlcvs._iter_timeframes())


def _make_ohlcvs(
    *,
    close: float = 100.0,
    reference: float = 100.0,
    band_low_1: float = 95.0,
    band_high_1: float = 105.0,
) -> OHLCVs:
    dates = pd.date_range("2024-02-19", periods=1, freq="1D")
    df = pd.DataFrame(
        {
            "open": [close],
            "high": [close + 1],
            "low": [close - 1],
            "close": [close],
            "volume": [1000.0],
        },
        index=dates,
    )
    ohlcvs = OHLCVs({TIMEFRAME: {BTC: df}})
    ohlcvs.add_column(BTC, TIMEFRAME, f"{SETUP_ID}_reference", [reference])
    ohlcvs.add_column(BTC, TIMEFRAME, f"{SETUP_ID}_band_low_1", [band_low_1])
    ohlcvs.add_column(BTC, TIMEFRAME, f"{SETUP_ID}_band_high_1", [band_high_1])

    _stand_on_first_candle(ohlcvs)

    return ohlcvs


def _account_state(
    *,
    open_orders: list[OrderProtocol] | None = None,
    positions: dict[Symbol, PositionSnapshot] | None = None,
    executions: list[Execution] | None = None,
    total_balance: float = 10_000.0,
) -> AccountSnapshots:
    snapshot = AccountSnapshot(
        account_name=ACCOUNT_NAME,
        balances={"USDT": Balance(locked=0.0, total=total_balance)},
        open_orders=[] if open_orders is None else open_orders,
        positions={} if positions is None else positions,
        executions=[] if executions is None else executions,
        executions_declared=True,
        margin_settings={
            BTC: MarginSettings(leverage=None, margin_mode=MarginMode.CROSS)
        },
        conversion_rates={BTC: 1.0},
    )
    return AccountSnapshots({ACCOUNT_NAME: snapshot})


PROFILE = {
    "symbol": "BTC/USDT:USDT",
    "timeframe": TIMEFRAME,
    "average_type": "SMA",
    "average_period": 100,
    "envelopes": [0.05],
    "stop_loss_pct": 0.04,
    "total_balance_ratio": 0.5,
    "leverage": 5.0,
    "margin_mode": "isolated",
}


def _directional_profile(**envelope_keys: object) -> dict[str, object]:
    return {
        key: value for key, value in PROFILE.items() if key != "envelopes"
    } | envelope_keys


PROFILE_LONG_ONLY = _directional_profile(long_envelopes=[0.05])
PROFILE_SHORT_ONLY = _directional_profile(short_envelopes=[0.05])
PROFILE_UNEQUAL_SIDES = _directional_profile(
    long_envelopes=[0.05, 0.10, 0.15], short_envelopes=[0.05]
)


_RUNG_ACTION = {"trigger": FuturesMarketOrderAction, "limit": FuturesLimitOrderAction}
_OTHER_ENTRY_ORDER = {"trigger": "limit", "limit": "trigger"}


@pytest.fixture
def make_strategy(
    make_strategy: Callable[..., EnvelopeStrategy], entry_order: str
) -> Callable[..., EnvelopeStrategy]:
    def _make(
        profiles: list[dict[str, Any]], **strategy_kwargs: Any
    ) -> EnvelopeStrategy:
        return make_strategy(profiles, entry_order=entry_order, **strategy_kwargs)

    return _make


@pytest.fixture
def strategy(make_strategy: Callable[..., EnvelopeStrategy]) -> EnvelopeStrategy:
    strategy = make_strategy([PROFILE])
    asyncio.run(strategy.setup(StrategyRequirements()))
    return strategy


def _rung_orders(
    bookkeeper: BookKeeper, entry_order: str
) -> list[BatchableOrderAction]:
    return [
        order
        for action in bookkeeper.list_actions()
        if isinstance(action, FuturesOrderBatchAction)
        for order in action.orders
        if isinstance(order, _RUNG_ACTION[entry_order])
    ]


def _level(order: BatchableOrderAction, entry_order: str) -> float | None:
    if not isinstance(order, _RUNG_ACTION[entry_order]):
        pytest.fail(f"{order!r} does not rest as a {entry_order} entry")
    if isinstance(order, FuturesLimitOrderAction):
        return order.price
    return order.trigger_price


def _modifications(bookkeeper: BookKeeper) -> list[OrderModification]:
    return [
        modification
        for action in bookkeeper.list_actions()
        if isinstance(action, FuturesOrderModifyAction)
        for modification in action.modifications
    ]


class TestMarginTargets:
    def test_each_symbol_declares_its_profiles_settings(self, make_strategy):
        requirements = StrategyRequirements()

        asyncio.run(make_strategy([PROFILE]).setup(requirements))

        assert requirements.account._get_all()[0].margin_targets == {
            BTC: MarginSettings(leverage=5.0, margin_mode=MarginMode.ISOLATED)
        }

    def test_a_drifted_symbol_books_no_margin_action(self, strategy):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(), _account_state(), TIMESTAMP, bookkeeper, [TIMEFRAME]
        )

        setters = [
            action
            for action in bookkeeper.list_actions()
            if isinstance(action, (SetLeverageAction, SetMarginModeAction))
        ]
        assert setters == []


class TestReconciliation:
    def test_flat_book_without_resting_orders(self, strategy):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(), _account_state(), TIMESTAMP, bookkeeper, [TIMEFRAME]
        )

        cancels = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, (CancelOrdersForSymbolAction, CancelOrderByIdAction))
        ]
        assert cancels == []
        assert _modifications(bookkeeper) == []

    def test_resting_rungs_are_reshaped_to_the_new_bands(self, strategy, entry_order):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            _account_state(
                open_orders=[
                    _fake_order(
                        "long-1", entry_order, OrderSide.BUY, level=96.0, rung=1
                    ),
                    _fake_order(
                        "short-1",
                        entry_order,
                        OrderSide.SELL,
                        level=104.0,
                        rung=1,
                    ),
                ]
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        reshaped = {
            modification.order_id: _level(modification.order, entry_order)
            for modification in _modifications(bookkeeper)
        }
        assert reshaped == {"long-1": 95.0, "short-1": 105.0}
        assert _rung_orders(bookkeeper, entry_order) == []

    def test_extra_resting_rungs_are_cancelled(self, strategy, entry_order):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            _account_state(
                open_orders=[
                    _fake_order(
                        "long-1", entry_order, OrderSide.BUY, level=96.0, rung=1
                    ),
                    _fake_order(
                        "long-2", entry_order, OrderSide.BUY, level=93.0, rung=2
                    ),
                    _fake_order(
                        "short-1",
                        entry_order,
                        OrderSide.SELL,
                        level=104.0,
                        rung=1,
                    ),
                ]
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        cancelled_ids = {
            a.order_id
            for a in bookkeeper.list_actions()
            if isinstance(a, CancelOrderByIdAction)
        }
        reshaped_ids = {m.order_id for m in _modifications(bookkeeper)}
        assert cancelled_ids == {"long-2"}
        assert reshaped_ids == {"long-1", "short-1"}

    def test_second_rung_on_the_same_band_is_cancelled(self, strategy, entry_order):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            _account_state(
                open_orders=[
                    _fake_order(
                        "long-1a", entry_order, OrderSide.BUY, level=96.0, rung=1
                    ),
                    _fake_order(
                        "long-1b", entry_order, OrderSide.BUY, level=94.0, rung=1
                    ),
                ]
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        cancelled_ids = {
            a.order_id
            for a in bookkeeper.list_actions()
            if isinstance(a, CancelOrderByIdAction)
        }
        reshaped_ids = {m.order_id for m in _modifications(bookkeeper)}
        assert cancelled_ids == {"long-1b"}
        assert reshaped_ids == {"long-1a"}

    def test_surviving_rung_keeps_its_own_band_while_the_others_refill(
        self, make_strategy, entry_order
    ):
        strategy = make_strategy([PROFILE_THREE_RUNGS])
        asyncio.run(strategy.setup(StrategyRequirements()))
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_three_rung_ohlcvs(),
            _account_state(
                open_orders=[
                    _fake_order(
                        "rung-3", entry_order, OrderSide.BUY, level=86.0, rung=3
                    ),
                ]
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        reshaped = {
            modification.order_id: _level(modification.order, entry_order)
            for modification in _modifications(bookkeeper)
        }
        placed_long_levels = sorted(
            _level(order, entry_order)
            for order in _rung_orders(bookkeeper, entry_order)
            if order.side == OrderSide.BUY
        )
        assert reshaped == {"rung-3": 85.0}
        assert placed_long_levels == [90.0, 95.0]

    def test_resting_rung_of_another_program_is_cancelled(self, strategy, entry_order):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            _account_state(
                open_orders=[
                    _fake_order("manual-1", entry_order, OrderSide.BUY, level=96.0)
                ]
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        cancelled_ids = {
            a.order_id
            for a in bookkeeper.list_actions()
            if isinstance(a, CancelOrderByIdAction)
        }
        assert cancelled_ids == {"manual-1"}
        assert _modifications(bookkeeper) == []

    def test_stray_resting_order_is_cancelled(self, strategy, entry_order):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            _account_state(
                open_orders=[
                    _fake_order(
                        "ghost-1", _OTHER_ENTRY_ORDER[entry_order], OrderSide.BUY
                    ),
                    _fake_order("sl-1", "stop-loss", OrderSide.SELL),
                    _fake_order("tp-1", "take-profit", OrderSide.SELL),
                ]
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        cancelled_ids = {
            a.order_id
            for a in bookkeeper.list_actions()
            if isinstance(a, CancelOrderByIdAction)
        }
        assert cancelled_ids == {"ghost-1"}


class TestRungShape:
    def test_long_rung_rests_at_low_band(self, strategy, entry_order):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(band_low_1=95.0),
            _account_state(),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        long_orders = [
            o for o in _rung_orders(bookkeeper, entry_order) if o.side == OrderSide.BUY
        ]
        assert len(long_orders) == 1
        assert _level(long_orders[0], entry_order) == 95.0

    def test_short_rung_rests_at_high_band(self, strategy, entry_order):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(band_high_1=105.0),
            _account_state(),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        short_orders = [
            o for o in _rung_orders(bookkeeper, entry_order) if o.side == OrderSide.SELL
        ]
        assert len(short_orders) == 1
        assert _level(short_orders[0], entry_order) == 105.0

    def test_long_rung_stop_loss_below_entry(self, strategy, entry_order):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(band_low_1=100.0),
            _account_state(),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        long_orders = [
            o for o in _rung_orders(bookkeeper, entry_order) if o.side == OrderSide.BUY
        ]
        assert long_orders[0].stop_loss.trigger_price == pytest.approx(
            100.0 * (1 - 0.04)
        )

    def test_short_rung_stop_loss_above_entry(self, strategy, entry_order):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(band_high_1=100.0),
            _account_state(),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        short_orders = [
            o for o in _rung_orders(bookkeeper, entry_order) if o.side == OrderSide.SELL
        ]
        assert short_orders[0].stop_loss.trigger_price == pytest.approx(
            100.0 * (1 + 0.04)
        )

    def test_rung_take_profit_at_reference(self, strategy, entry_order):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(reference=110.0),
            _account_state(),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        take_profit_by_side = {
            order.side: order.take_profit.trigger_price
            for order in _rung_orders(bookkeeper, entry_order)
        }
        assert take_profit_by_side == {
            OrderSide.BUY: pytest.approx(110.0),
            OrderSide.SELL: pytest.approx(110.0),
        }

    def test_no_rung_rests_as_the_other_entry_order(self, strategy, entry_order):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(), _account_state(), TIMESTAMP, bookkeeper, [TIMEFRAME]
        )

        assert _rung_orders(bookkeeper, _OTHER_ENTRY_ORDER[entry_order]) == []

    def test_rung_carries_its_tag(self, strategy, entry_order):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(), _account_state(), TIMESTAMP, bookkeeper, [TIMEFRAME]
        )

        tags = {order.tag for order in _rung_orders(bookkeeper, entry_order)}
        assert tags == {"r1"}

    def test_rungs_are_sized_from_the_balance(self, strategy, entry_order):
        total_balance = 10_000.0
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            _account_state(total_balance=total_balance),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        quantity_by_level = {
            _level(order, entry_order): order.quantity
            for order in _rung_orders(bookkeeper, entry_order)
        }
        notional = total_balance * PROFILE["total_balance_ratio"]
        assert quantity_by_level == {
            95.0: pytest.approx(notional / 95.0),
            105.0: pytest.approx(notional / 105.0),
        }


class TestDirectionalProfiles:
    def test_long_envelopes_alone_emits_only_long_rungs(
        self, make_strategy, entry_order
    ):
        strategy = make_strategy([PROFILE_LONG_ONLY])
        asyncio.run(strategy.setup(StrategyRequirements()))
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(), _account_state(), TIMESTAMP, bookkeeper, [TIMEFRAME]
        )

        orders = _rung_orders(bookkeeper, entry_order)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY

    def test_short_envelopes_alone_emits_only_short_rungs(
        self, make_strategy, entry_order
    ):
        strategy = make_strategy([PROFILE_SHORT_ONLY])
        asyncio.run(strategy.setup(StrategyRequirements()))
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(), _account_state(), TIMESTAMP, bookkeeper, [TIMEFRAME]
        )

        orders = _rung_orders(bookkeeper, entry_order)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL

    def test_unequal_side_counts_emit_their_own_number_of_rungs(
        self, make_strategy, entry_order
    ):
        strategy = make_strategy([PROFILE_UNEQUAL_SIDES])
        asyncio.run(strategy.setup(StrategyRequirements()))
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_three_rung_ohlcvs(),
            _account_state(),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        orders = _rung_orders(bookkeeper, entry_order)
        long_orders = [o for o in orders if o.side == OrderSide.BUY]
        short_orders = [o for o in orders if o.side == OrderSide.SELL]
        assert len(long_orders) == 3
        assert len(short_orders) == 1

    def test_unequal_side_counts_size_rungs_from_their_own_side(
        self, make_strategy, entry_order
    ):
        strategy = make_strategy([PROFILE_UNEQUAL_SIDES])
        asyncio.run(strategy.setup(StrategyRequirements()))
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_three_rung_ohlcvs(),
            _account_state(),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        quantity_by_level = {
            _level(order, entry_order): order.quantity
            for order in _rung_orders(bookkeeper, entry_order)
        }
        balance = 10_000.0
        long_ratio = PROFILE_UNEQUAL_SIDES["total_balance_ratio"] / 3
        short_ratio = PROFILE_UNEQUAL_SIDES["total_balance_ratio"] / 1
        assert quantity_by_level[95.0] == pytest.approx(balance * long_ratio / 95.0)
        assert quantity_by_level[90.0] == pytest.approx(balance * long_ratio / 90.0)
        assert quantity_by_level[85.0] == pytest.approx(balance * long_ratio / 85.0)
        assert quantity_by_level[105.0] == pytest.approx(balance * short_ratio / 105.0)

    def test_a_notional_splits_over_the_rungs_of_each_side(
        self, make_strategy, entry_order
    ):
        profile = {**PROFILE_UNEQUAL_SIDES, "notional": 300.0}
        del profile["total_balance_ratio"]
        strategy = make_strategy([profile])
        asyncio.run(strategy.setup(StrategyRequirements()))
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_three_rung_ohlcvs(),
            _account_state(),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        quantity_by_level = {
            _level(order, entry_order): order.quantity
            for order in _rung_orders(bookkeeper, entry_order)
        }
        assert quantity_by_level[90.0] == pytest.approx(300.0 / 3 / 90.0)
        assert quantity_by_level[105.0] == pytest.approx(300.0 / 105.0)

    def test_an_empty_balance_books_rungs_sized_to_nothing(
        self, make_strategy, entry_order
    ):
        strategy = make_strategy([PROFILE_UNEQUAL_SIDES])
        asyncio.run(strategy.setup(StrategyRequirements()))
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_three_rung_ohlcvs(),
            _account_state(total_balance=0.0),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        assert [order.quantity for order in _rung_orders(bookkeeper, entry_order)] == [
            0.0
        ] * 4


class TestSkipConditions:
    def test_untriggered_timeframe_skips_profile(self, strategy, entry_order):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(), _account_state(), TIMESTAMP, bookkeeper, ["4h"]
        )

        assert _rung_orders(bookkeeper, entry_order) == []


class TestNanLevels:
    def test_nan_reference_cancels_orders_and_emits_no_rungs(
        self, strategy, entry_order
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(
                reference=float("nan"),
                band_low_1=float("nan"),
                band_high_1=float("nan"),
            ),
            _account_state(),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        cancels = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, CancelOrdersForSymbolAction)
        ]
        assert len(cancels) == 1
        assert cancels[0].symbol == BTC
        assert _rung_orders(bookkeeper, entry_order) == []

    def test_nan_long_band_alone_emits_no_rungs(self, strategy, entry_order):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(band_low_1=float("nan")),
            _account_state(),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        assert _rung_orders(bookkeeper, entry_order) == []

    def test_nan_short_band_alone_emits_no_rungs(self, strategy, entry_order):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(band_high_1=float("nan")),
            _account_state(),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        assert _rung_orders(bookkeeper, entry_order) == []


def _fake_order(
    order_id: str,
    order_type: str,
    side: OrderSide,
    symbol: Symbol = BTC,
    level: float | None = None,
    rung: int | None = None,
) -> Mock:
    order = Mock(spec=OrderProtocol)
    order.order_id = order_id
    order.kind = order_type
    order.symbol = symbol
    order.side = side
    order.trigger_price = level if order_type == "trigger" else None
    order.price = level if order_type == "limit" else None
    order.client_order_id = None if rung is None else f"{_FAKE_PROCESS_TAG}-r{rung}"
    return order


def _open_long_position(
    quantity: float = 1.0, entry_price: float = 100.0
) -> PositionSnapshot:
    return PositionSnapshot(
        symbol=BTC,
        side=PositionSide.LONG,
        quantity=quantity,
        average_entry_price=entry_price,
        entry_time=TIMESTAMP,
        leverage=5.0,
        liquidation_price=entry_price * 0.5,
    )


def _open_short_position(
    quantity: float = 1.0, entry_price: float = 100.0
) -> PositionSnapshot:
    return PositionSnapshot(
        symbol=BTC,
        side=PositionSide.SHORT,
        quantity=quantity,
        average_entry_price=entry_price,
        entry_time=TIMESTAMP,
        leverage=5.0,
        liquidation_price=entry_price * 1.5,
    )


PROFILE_THREE_RUNGS = {
    **PROFILE,
    "envelopes": [0.05, 0.10, 0.15],
}
PROFILE_THREE_SHORT_ONE_LONG = _directional_profile(
    long_envelopes=[0.05], short_envelopes=[0.05, 0.10, 0.15]
)


def _make_three_rung_ohlcvs(
    *,
    reference: float = 100.0,
    band_lows: tuple[float, float, float] = (95.0, 90.0, 85.0),
    band_highs: tuple[float, float, float] = (105.0, 110.0, 115.0),
) -> OHLCVs:
    dates = pd.date_range("2024-02-19", periods=1, freq="1D")
    df = pd.DataFrame(
        {
            "open": [reference],
            "high": [reference + 1],
            "low": [reference - 1],
            "close": [reference],
            "volume": [1000.0],
        },
        index=dates,
    )
    ohlcvs = OHLCVs({TIMEFRAME: {BTC: df}})
    ohlcvs.add_column(BTC, TIMEFRAME, f"{SETUP_ID}_reference", [reference])
    for index, (low, high) in enumerate(zip(band_lows, band_highs), start=1):
        ohlcvs.add_column(BTC, TIMEFRAME, f"{SETUP_ID}_band_low_{index}", [low])
        ohlcvs.add_column(BTC, TIMEFRAME, f"{SETUP_ID}_band_high_{index}", [high])
    _stand_on_first_candle(ohlcvs)
    return ohlcvs


class TestOpenPosition:
    def test_position_side_rung_is_reshaped(self, strategy, entry_order):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            _account_state(
                positions={BTC: _open_long_position()},
                open_orders=[
                    _fake_order(
                        "rung-1", entry_order, OrderSide.BUY, level=96.0, rung=1
                    ),
                    _fake_order("tp-1", "take-profit", OrderSide.SELL),
                    _fake_order("sl-1", "stop-loss", OrderSide.SELL),
                ],
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        cancels = [
            a for a in bookkeeper.list_actions() if isinstance(a, CancelOrderByIdAction)
        ]
        reshaped = {
            modification.order_id: _level(modification.order, entry_order)
            for modification in _modifications(bookkeeper)
        }
        assert cancels == []
        assert reshaped == {"rung-1": 95.0}

    def test_opposite_side_rung_is_cancelled(self, strategy, entry_order):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            _account_state(
                positions={BTC: _open_long_position()},
                open_orders=[
                    _fake_order(
                        "short-1",
                        entry_order,
                        OrderSide.SELL,
                        level=104.0,
                        rung=1,
                    ),
                ],
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        cancelled_ids = {
            a.order_id
            for a in bookkeeper.list_actions()
            if isinstance(a, CancelOrderByIdAction)
        }
        assert cancelled_ids == {"short-1"}

    def test_emits_update_take_profit_at_reference(self, strategy):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(reference=110.0),
            _account_state(positions={BTC: _open_long_position()}),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        tp_updates = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, UpdatePositionTakeProfitAction)
        ]
        assert len(tp_updates) == 1
        assert tp_updates[0].symbol == BTC
        assert tp_updates[0].trigger_price == pytest.approx(110.0)

    def test_emits_update_stop_loss_from_average_entry_for_long(self, strategy):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            _account_state(positions={BTC: _open_long_position(entry_price=98.0)}),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        sl_updates = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, UpdatePositionStopLossAction)
        ]
        assert len(sl_updates) == 1
        assert sl_updates[0].trigger_price == pytest.approx(98.0 * (1 - 0.04))

    def test_emits_update_stop_loss_from_average_entry_for_short(self, strategy):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            _account_state(positions={BTC: _open_short_position(entry_price=102.0)}),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        sl_updates = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, UpdatePositionStopLossAction)
        ]
        assert len(sl_updates) == 1
        assert sl_updates[0].trigger_price == pytest.approx(102.0 * (1 + 0.04))

    def test_long_position_reshapes_only_unfilled_long_rungs(
        self, make_strategy, entry_order
    ):
        strategy = make_strategy([PROFILE_THREE_RUNGS])
        asyncio.run(strategy.setup(StrategyRequirements()))
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_three_rung_ohlcvs(),
            _account_state(
                positions={BTC: _open_long_position()},
                open_orders=[
                    _fake_order(
                        "rung-2", entry_order, OrderSide.BUY, level=91.0, rung=2
                    ),
                    _fake_order(
                        "rung-3", entry_order, OrderSide.BUY, level=86.0, rung=3
                    ),
                ],
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        reshaped = {
            modification.order_id: _level(modification.order, entry_order)
            for modification in _modifications(bookkeeper)
        }
        assert reshaped == {"rung-2": 90.0, "rung-3": 85.0}
        assert _rung_orders(bookkeeper, entry_order) == []

    def test_short_position_reshapes_only_unfilled_short_rungs(
        self, make_strategy, entry_order
    ):
        strategy = make_strategy([PROFILE_THREE_RUNGS])
        asyncio.run(strategy.setup(StrategyRequirements()))
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_three_rung_ohlcvs(),
            _account_state(
                positions={BTC: _open_short_position()},
                open_orders=[
                    _fake_order(
                        "rung-2", entry_order, OrderSide.SELL, level=109.0, rung=2
                    ),
                    _fake_order(
                        "rung-3", entry_order, OrderSide.SELL, level=114.0, rung=3
                    ),
                ],
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        reshaped = {
            modification.order_id: _level(modification.order, entry_order)
            for modification in _modifications(bookkeeper)
        }
        assert reshaped == {"rung-2": 110.0, "rung-3": 115.0}
        assert _rung_orders(bookkeeper, entry_order) == []

    def test_long_position_reshape_uses_the_long_sides_own_count(
        self, make_strategy, entry_order
    ):
        strategy = make_strategy([PROFILE_UNEQUAL_SIDES])
        asyncio.run(strategy.setup(StrategyRequirements()))
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_three_rung_ohlcvs(),
            _account_state(
                positions={BTC: _open_long_position()},
                open_orders=[
                    _fake_order(
                        "rung-2", entry_order, OrderSide.BUY, level=91.0, rung=2
                    ),
                    _fake_order(
                        "rung-3", entry_order, OrderSide.BUY, level=86.0, rung=3
                    ),
                ],
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        reshaped = {
            modification.order_id: _level(modification.order, entry_order)
            for modification in _modifications(bookkeeper)
        }
        assert reshaped == {"rung-2": 90.0, "rung-3": 85.0}
        assert _rung_orders(bookkeeper, entry_order) == []

    def test_short_position_reshape_uses_the_short_sides_own_count(
        self, make_strategy, entry_order
    ):
        strategy = make_strategy([PROFILE_THREE_SHORT_ONE_LONG])
        asyncio.run(strategy.setup(StrategyRequirements()))
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_three_rung_ohlcvs(),
            _account_state(
                positions={BTC: _open_short_position()},
                open_orders=[
                    _fake_order(
                        "rung-2", entry_order, OrderSide.SELL, level=109.0, rung=2
                    ),
                    _fake_order(
                        "rung-3", entry_order, OrderSide.SELL, level=114.0, rung=3
                    ),
                ],
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        reshaped = {
            modification.order_id: _level(modification.order, entry_order)
            for modification in _modifications(bookkeeper)
        }
        assert reshaped == {"rung-2": 110.0, "rung-3": 115.0}
        assert _rung_orders(bookkeeper, entry_order) == []

    def test_nan_levels_cancel_only_rungs_and_skip_updates(self, strategy, entry_order):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(
                reference=float("nan"),
                band_low_1=float("nan"),
                band_high_1=float("nan"),
            ),
            _account_state(
                positions={BTC: _open_long_position()},
                open_orders=[
                    _fake_order(
                        "rung-1", entry_order, OrderSide.BUY, level=96.0, rung=1
                    ),
                    _fake_order("sl-1", "stop-loss", OrderSide.SELL),
                ],
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        cancelled_ids = {
            a.order_id
            for a in bookkeeper.list_actions()
            if isinstance(a, CancelOrderByIdAction)
        }
        updates = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(
                a, (UpdatePositionStopLossAction, UpdatePositionTakeProfitAction)
            )
        ]
        assert cancelled_ids == {"rung-1"}
        assert updates == []
        assert _rung_orders(bookkeeper, entry_order) == []

    def test_does_not_emit_cancel_all_for_symbol(self, strategy):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            _account_state(positions={BTC: _open_long_position()}),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        cancel_all = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, CancelOrdersForSymbolAction)
        ]
        assert cancel_all == []


class TestCancelReplaceMode:
    @pytest.fixture
    def strategy(
        self, make_strategy: Callable[..., EnvelopeStrategy]
    ) -> EnvelopeStrategy:
        strategy = make_strategy([PROFILE], update_mode="cancel_replace")
        asyncio.run(strategy.setup(StrategyRequirements()))
        return strategy

    def test_flat_book_cancels_all_orders_for_the_symbol(self, strategy):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(), _account_state(), TIMESTAMP, bookkeeper, [TIMEFRAME]
        )

        cancels = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, CancelOrdersForSymbolAction)
        ]
        assert len(cancels) == 1
        assert cancels[0].symbol == BTC

    def test_resting_rungs_are_replaced_not_reshaped(self, strategy, entry_order):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            _account_state(
                open_orders=[
                    _fake_order(
                        "long-1", entry_order, OrderSide.BUY, level=96.0, rung=1
                    ),
                ]
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        orders = _rung_orders(bookkeeper, entry_order)
        assert _modifications(bookkeeper) == []
        assert {o.side: _level(o, entry_order) for o in orders} == {
            OrderSide.BUY: 95.0,
            OrderSide.SELL: 105.0,
        }

    def test_open_position_cancels_pending_rungs_by_id(self, strategy, entry_order):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            _account_state(
                positions={BTC: _open_long_position()},
                open_orders=[
                    _fake_order("rung-1", entry_order, OrderSide.BUY),
                    _fake_order("tp-1", "take-profit", OrderSide.SELL),
                    _fake_order("sl-1", "stop-loss", OrderSide.SELL),
                ],
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        cancelled_ids = {
            a.order_id
            for a in bookkeeper.list_actions()
            if isinstance(a, CancelOrderByIdAction)
        }
        assert cancelled_ids == {"rung-1"}

    def test_long_position_re_emits_only_unfilled_long_rungs(
        self, make_strategy, entry_order
    ):
        strategy = make_strategy([PROFILE_THREE_RUNGS], update_mode="cancel_replace")
        asyncio.run(strategy.setup(StrategyRequirements()))
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_three_rung_ohlcvs(),
            _account_state(
                positions={BTC: _open_long_position()},
                open_orders=[
                    _fake_order("rung-2", entry_order, OrderSide.BUY),
                    _fake_order("rung-3", entry_order, OrderSide.BUY),
                ],
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        entry_orders = [
            o for o in _rung_orders(bookkeeper, entry_order) if not o.reduce_only
        ]
        assert len(entry_orders) == 2
        assert {o.side for o in entry_orders} == {OrderSide.BUY}
        assert {_level(o, entry_order) for o in entry_orders} == {90.0, 85.0}

    def test_open_position_does_not_emit_cancel_all_for_symbol(self, strategy):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            _account_state(positions={BTC: _open_long_position()}),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        cancel_all = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, CancelOrdersForSymbolAction)
        ]
        assert cancel_all == []


def _long_sl_fill(timestamp: datetime) -> Execution:
    return Execution(
        execution_id="sl-long",
        order_id="sl-long",
        symbol=BTC,
        side=OrderSide.SELL,
        price=100.0,
        quantity=1.0,
        timestamp=timestamp,
        kind="stop-loss",
    )


def _short_sl_fill(timestamp: datetime) -> Execution:
    return Execution(
        execution_id="sl-short",
        order_id="sl-short",
        symbol=BTC,
        side=OrderSide.BUY,
        price=100.0,
        quantity=1.0,
        timestamp=timestamp,
        kind="stop-loss",
    )


class TestReentryGating:
    def test_long_locked_suppresses_long_rungs(self, strategy, entry_order):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(close=95.0, reference=100.0),
            _account_state(executions=[_long_sl_fill(TIMESTAMP)]),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        orders = _rung_orders(bookkeeper, entry_order)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL

    def test_short_locked_suppresses_short_rungs(self, strategy, entry_order):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(close=105.0, reference=100.0),
            _account_state(executions=[_short_sl_fill(TIMESTAMP)]),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        orders = _rung_orders(bookkeeper, entry_order)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY

    def test_both_locked_suppresses_all_rungs(self, strategy, entry_order):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(close=100.0, reference=100.0),
            _account_state(
                executions=[_long_sl_fill(TIMESTAMP), _short_sl_fill(TIMESTAMP)]
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        assert _rung_orders(bookkeeper, entry_order) == []

    def test_open_position_with_own_side_locked_skips_re_emission(
        self, make_strategy, entry_order
    ):
        strategy = make_strategy([PROFILE_THREE_RUNGS])
        asyncio.run(strategy.setup(StrategyRequirements()))
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_three_rung_ohlcvs(reference=100.0),
            _account_state(
                positions={BTC: _open_long_position()},
                open_orders=[
                    _fake_order(
                        "rung-2", entry_order, OrderSide.BUY, level=91.0, rung=2
                    ),
                    _fake_order(
                        "rung-3", entry_order, OrderSide.BUY, level=86.0, rung=3
                    ),
                ],
                executions=[_long_sl_fill(TIMESTAMP)],
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        entry_orders = [
            o for o in _rung_orders(bookkeeper, entry_order) if not o.reduce_only
        ]
        assert entry_orders == []
        assert _modifications(bookkeeper) == []

    def test_open_position_with_own_side_locked_still_updates_sl_tp(
        self, make_strategy
    ):
        strategy = make_strategy([PROFILE_THREE_RUNGS])
        asyncio.run(strategy.setup(StrategyRequirements()))
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_three_rung_ohlcvs(reference=100.0),
            _account_state(
                positions={BTC: _open_long_position()},
                executions=[_long_sl_fill(TIMESTAMP)],
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        sl_updates = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, UpdatePositionStopLossAction)
        ]
        tp_updates = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, UpdatePositionTakeProfitAction)
        ]
        assert len(sl_updates) == 1
        assert len(tp_updates) == 1


class TestFillReasons:
    @staticmethod
    def _declared_describer(
        strategy: EnvelopeStrategy,
    ) -> Callable[[Execution], str | None]:
        requirements = StrategyRequirements()
        asyncio.run(strategy.setup(requirements))
        describer = requirements.account._get_all()[0].describe_fills
        assert describer is not None
        return describer

    def test_names_the_rung_its_tag_claims(self, make_strategy, entry_order):
        strategy = make_strategy([PROFILE_THREE_RUNGS])
        fill = _fired_entry(f"{_FAKE_PROCESS_TAG}-r2", kind=entry_order)

        reason = self._declared_describer(strategy)(fill)

        assert reason == "entry rung 2 of 3"

    def test_counts_the_fills_own_side_when_sides_are_unequal(
        self, make_strategy, entry_order
    ):
        strategy = make_strategy([PROFILE_UNEQUAL_SIDES])
        long_fill = _fired_entry(
            f"{_FAKE_PROCESS_TAG}-r2", side=OrderSide.BUY, kind=entry_order
        )
        short_fill = _fired_entry(
            f"{_FAKE_PROCESS_TAG}-r1", side=OrderSide.SELL, kind=entry_order
        )
        describer = self._declared_describer(strategy)

        assert describer(long_fill) == "entry rung 2 of 3"
        assert describer(short_fill) == "entry rung 1 of 1"

    def test_fill_of_another_program(self, strategy, entry_order):
        fill = _fired_entry("t1785908301620", kind=entry_order)

        reason = self._declared_describer(strategy)(fill)

        assert reason is None

    def test_fill_without_a_client_order_id(self, strategy, entry_order):
        fill = _fired_entry(None, kind=entry_order)

        reason = self._declared_describer(strategy)(fill)

        assert reason is None

    def test_take_profit_means_the_moving_average_was_reached(self, strategy):
        fill = _venue_fired_exit("take-profit")

        reason = self._declared_describer(strategy)(fill)

        assert reason == "moving average reached"

    def test_stop_loss_is_left_to_its_own_name(self, strategy):
        fill = _venue_fired_exit("stop-loss")

        reason = self._declared_describer(strategy)(fill)

        assert reason is None

    def test_fill_of_the_other_entry_order(self, strategy, entry_order):
        fill = _fired_entry(
            f"{_FAKE_PROCESS_TAG}-r1", kind=_OTHER_ENTRY_ORDER[entry_order]
        )

        reason = self._declared_describer(strategy)(fill)

        assert reason is None


def _venue_fired_exit(kind: OrderType) -> Execution:
    return Execution(
        execution_id="fired-2",
        order_id="fired-2",
        symbol=BTC,
        side=OrderSide.SELL,
        price=105.0,
        quantity=0.5,
        timestamp=TIMESTAMP + timedelta(hours=6),
        kind=kind,
    )


def _fired_entry(
    client_order_id: str | None,
    kind: OrderType,
    side: OrderSide = OrderSide.BUY,
) -> Execution:
    return Execution(
        execution_id="fired-1",
        order_id="fired-1",
        symbol=BTC,
        side=side,
        price=95.0,
        quantity=0.5,
        timestamp=TIMESTAMP + timedelta(hours=5),
        kind=kind,
        client_order_id=client_order_id,
    )
