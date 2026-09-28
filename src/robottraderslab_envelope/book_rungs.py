from collections.abc import Callable

from robottraderslab.strategies import (
    BookKeeper,
    OrderProtocol,
    OrderSide,
)
from robottraderslab.strategies.futures import (
    BatchableOrderAction,
    FuturesAccount,
    OrderModification,
)

from .profile_config import ProfileConfig
from .profile_snapshot import ProfileSnapshot
from .rungs import WantedRung, rung_band_index

type RungUpdater = Callable[
    [
        FuturesAccount,
        ProfileConfig,
        ProfileSnapshot,
        list[WantedRung],
        BookKeeper,
    ],
    None,
]


def book_cancel_and_place(
    account: FuturesAccount,
    profile: ProfileConfig,
    profile_snapshot: ProfileSnapshot,
    wanted: list[WantedRung],
    bookkeeper: BookKeeper,
) -> None:
    """Drop the resting rungs and place the wanted set once they are gone."""
    symbol = profile.symbol
    if profile_snapshot.position is None:
        cancels = [bookkeeper.add(account.cancel_orders(symbol), after=())]
    else:
        cancels = [
            bookkeeper.add(
                account.cancel_order(symbol, rung.order_id),
                after=(),
            )
            for rung in profile_snapshot.pending_rungs
        ]

    orders: list[BatchableOrderAction] = [rung.order for rung in wanted]
    if orders:
        bookkeeper.add(account.place_orders(orders), after=cancels)


def book_modify_in_place(
    account: FuturesAccount,
    profile: ProfileConfig,
    profile_snapshot: ProfileSnapshot,
    wanted: list[WantedRung],
    bookkeeper: BookKeeper,
) -> None:
    """Move each resting rung to its own band, so the book is never empty."""
    symbol = profile.symbol
    holding = profile_snapshot.position is not None

    modifications: list[OrderModification] = []
    cancel_ids: list[str] = [] if holding else list(profile_snapshot.stray_order_ids)
    missing: list[BatchableOrderAction] = []
    for side in (OrderSide.BUY, OrderSide.SELL):
        resting, unclaimed = _resting_by_band(profile_snapshot.pending_rungs, side)
        cancel_ids.extend(unclaimed)
        for rung in wanted:
            if rung.order.side != side:
                continue
            claimed = resting.pop(rung.band_index, None)
            if claimed is None:
                missing.append(rung.order)
            else:
                modifications.append(
                    OrderModification(order_id=claimed.order_id, order=rung.order)
                )
        cancel_ids.extend(rung.order_id for rung in resting.values())

    if modifications:
        bookkeeper.add(account.modify_orders(modifications), after=())
    cancels = [
        bookkeeper.add(account.cancel_order(symbol, order_id), after=())
        for order_id in cancel_ids
    ]
    if missing:
        bookkeeper.add(account.place_orders(missing), after=cancels)


def book_cancel_entries(
    account: FuturesAccount,
    profile: ProfileConfig,
    profile_snapshot: ProfileSnapshot,
    bookkeeper: BookKeeper,
) -> None:
    """Clear the resting rungs without placing anything in their place."""
    if profile_snapshot.position is None:
        bookkeeper.add(account.cancel_orders(profile.symbol), after=())
        return
    for rung in profile_snapshot.pending_rungs:
        bookkeeper.add(
            account.cancel_order(profile.symbol, rung.order_id),
            after=(),
        )


def _resting_by_band(
    pending_rungs: list[OrderProtocol], side: OrderSide
) -> tuple[dict[int, OrderProtocol], list[str]]:
    resting: dict[int, OrderProtocol] = {}
    unclaimed: list[str] = []
    for rung in pending_rungs:
        if rung.side != side:
            continue
        band_index = rung_band_index(rung.client_order_id)
        if band_index is None or band_index in resting:
            unclaimed.append(rung.order_id)
        else:
            resting[band_index] = rung
    return resting, unclaimed
