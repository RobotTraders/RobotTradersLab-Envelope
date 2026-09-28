import math
from dataclasses import dataclass
from typing import Self

from robottraderslab.strategies import (
    AccountSnapshot,
    OHLCVs,
    OrderProtocol,
    OrderType,
    PositionSide,
)
from robottraderslab.strategies.futures import PositionSnapshot

from .entry_order import EntryOrder
from .profile_config import ProfileConfig

_PROTECTION_ORDER_TYPES = (OrderType.STOP_LOSS, OrderType.TAKE_PROFIT)


@dataclass(frozen=True, slots=True)
class ProfileSnapshot:
    """Per-profile view of the current candle: market data plus exchange state."""

    reference: float
    bands: dict[PositionSide, list[float]]
    pending_rungs: list[OrderProtocol]
    stray_order_ids: list[str]
    position: PositionSnapshot | None

    @classmethod
    def build(
        cls,
        profile: ProfileConfig,
        ohlcvs: OHLCVs,
        account_snapshot: AccountSnapshot,
        entry_order: EntryOrder,
    ) -> Self:
        """Read one profile's levels and resting orders for the current candle."""
        symbol = profile.symbol
        timeframe = profile.timeframe

        pending_rungs: list[OrderProtocol] = []
        stray_order_ids: list[str] = []
        for order in account_snapshot.open_orders(symbol):
            if entry_order.matches(order.kind):
                pending_rungs.append(order)
            elif order.kind not in _PROTECTION_ORDER_TYPES:
                stray_order_ids.append(order.order_id)

        return cls(
            reference=ohlcvs.current(symbol, timeframe, profile.reference_column),
            bands={
                side: [ohlcvs.current(symbol, timeframe, column) for column in columns]
                for side, columns in profile.band_columns.items()
            },
            pending_rungs=pending_rungs,
            stray_order_ids=stray_order_ids,
            position=account_snapshot.position(symbol),
        )

    @property
    def has_nan_levels(self) -> bool:
        return math.isnan(self.reference) or any(
            math.isnan(price) for prices in self.bands.values() for price in prices
        )
