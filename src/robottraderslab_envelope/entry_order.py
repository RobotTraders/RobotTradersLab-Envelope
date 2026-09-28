from robottraderslab.strategies import OrderType
from robottraderslab.strategies.futures import FuturesOrderBuilder


class LimitEntryOrder:
    def matches(self, order_kind: OrderType | None) -> bool:
        return order_kind == OrderType.LIMIT

    def rest_at_band(
        self, entry: FuturesOrderBuilder, band_price: float
    ) -> FuturesOrderBuilder:
        return entry.limit(price=band_price)


class TriggerEntryOrder:
    def matches(self, order_kind: OrderType | None) -> bool:
        return order_kind == OrderType.TRIGGER

    def rest_at_band(
        self, entry: FuturesOrderBuilder, band_price: float
    ) -> FuturesOrderBuilder:
        return entry.trigger(price=band_price)


type EntryOrder = LimitEntryOrder | TriggerEntryOrder
