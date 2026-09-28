from unittest.mock import Mock

import pandas as pd
import pytest
from robottraderslab_envelope.entry_order import (
    EntryOrder,
    LimitEntryOrder,
    TriggerEntryOrder,
)
from robottraderslab_envelope.profile_config import AverageType, ProfileConfig
from robottraderslab_envelope.profile_snapshot import ProfileSnapshot

from robottraderslab import Symbol
from robottraderslab.strategies import (
    AccountSnapshot,
    Balance,
    OHLCVs,
    OrderProtocol,
    OrderSide,
    PositionSide,
)
from robottraderslab.strategies.futures import (
    MarginMode,
    MarginSettings,
    TotalBalanceRatio,
)

BTC = Symbol.create("BTC/USDT:USDT")
TIMEFRAME = "1h"


def _stand_on_first_candle(ohlcvs: OHLCVs) -> None:
    """Set the iteration cursor so `current` and `signal` read the first candle."""
    next(ohlcvs._iter_timeframes())


def _order(order_id: str, order_type: str, trigger_price: float | None = None) -> Mock:
    order = Mock(spec=OrderProtocol)
    order.order_id = order_id
    order.symbol = BTC
    order.kind = order_type
    order.side = OrderSide.BUY
    order.trigger_price = trigger_price
    order.client_order_id = None
    return order


def _build(
    orders: list[Mock],
    profile: ProfileConfig,
    ohlcvs: OHLCVs,
    entry_order: EntryOrder = TriggerEntryOrder(),
) -> ProfileSnapshot:
    account_snapshot = AccountSnapshot(
        account_name="test",
        balances={"USDT": Balance(locked=0.0, total=10_000.0)},
        open_orders=orders,
        positions={},
        margin_settings={
            BTC: MarginSettings(leverage=None, margin_mode=MarginMode.CROSS)
        },
        conversion_rates={BTC: 1.0},
    )
    return ProfileSnapshot.build(profile, ohlcvs, account_snapshot, entry_order)


@pytest.fixture
def profile() -> ProfileConfig:
    return ProfileConfig(
        sizing=TotalBalanceRatio(1.0),
        symbol=str(BTC),
        timeframe=TIMEFRAME,
        average_type=AverageType.SMA,
        average_period=100,
        envelopes=[0.05],
        stop_loss_pct=0.04,
    )


@pytest.fixture
def ohlcvs(profile: ProfileConfig) -> OHLCVs:
    frame = pd.DataFrame(
        {"close": [100.0]}, index=pd.date_range("2024-01-01", periods=1, freq="h")
    )
    ohlcvs = OHLCVs({TIMEFRAME: {BTC: frame}})
    ohlcvs.add_column(BTC, TIMEFRAME, profile.reference_column, [100.0])
    ohlcvs.add_column(
        BTC, TIMEFRAME, profile.band_columns[PositionSide.LONG][0], [95.0]
    )
    ohlcvs.add_column(
        BTC, TIMEFRAME, profile.band_columns[PositionSide.SHORT][0], [105.0]
    )
    _stand_on_first_candle(ohlcvs)
    return ohlcvs


class TestOrderSorting:
    def test_trigger_order_is_a_rung(self, profile, ohlcvs):
        snapshot = _build([_order("1", "trigger", 95.0)], profile, ohlcvs)

        assert [rung.order_id for rung in snapshot.pending_rungs] == ["1"]
        assert snapshot.stray_order_ids == []

    def test_trigger_order_without_a_trigger_price_is_still_a_rung(
        self, profile, ohlcvs
    ):
        snapshot = _build([_order("1", "trigger")], profile, ohlcvs)

        assert [rung.order_id for rung in snapshot.pending_rungs] == ["1"]
        assert snapshot.stray_order_ids == []

    def test_protection_orders_are_neither_rung_nor_stray(self, profile, ohlcvs):
        orders = [_order("1", "stop-loss"), _order("2", "take-profit")]

        snapshot = _build(orders, profile, ohlcvs)

        assert snapshot.pending_rungs == []
        assert snapshot.stray_order_ids == []

    def test_other_order_types_are_strays(self, profile, ohlcvs):
        snapshot = _build([_order("1", "limit")], profile, ohlcvs)

        assert snapshot.pending_rungs == []
        assert snapshot.stray_order_ids == ["1"]

    def test_limit_order_is_a_rung_when_rungs_rest_as_limits(self, profile, ohlcvs):
        snapshot = _build([_order("1", "limit")], profile, ohlcvs, LimitEntryOrder())

        assert [rung.order_id for rung in snapshot.pending_rungs] == ["1"]
        assert snapshot.stray_order_ids == []

    def test_trigger_order_is_a_stray_when_rungs_rest_as_limits(self, profile, ohlcvs):
        snapshot = _build(
            [_order("1", "trigger", 95.0)], profile, ohlcvs, LimitEntryOrder()
        )

        assert snapshot.pending_rungs == []
        assert snapshot.stray_order_ids == ["1"]


class TestLevels:
    def test_reads_reference_and_bands_for_the_current_bar(self, profile, ohlcvs):
        snapshot = _build([], profile, ohlcvs)

        assert snapshot.reference == 100.0
        assert snapshot.bands == {
            PositionSide.LONG: [95.0],
            PositionSide.SHORT: [105.0],
        }
        assert snapshot.has_nan_levels is False
