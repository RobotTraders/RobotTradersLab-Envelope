from dataclasses import dataclass

from robottraderslab.strategies import AccountSnapshot, OrderSide, PositionSide, tag_of
from robottraderslab.strategies.futures import (
    BatchableOrderAction,
    FuturesAccount,
    PositionSnapshot,
)

from .entry_order import EntryOrder
from .position_protection import stop_loss_trigger_price
from .profile_config import ProfileConfig
from .profile_snapshot import ProfileSnapshot
from .reentry_guard import ReentryState

_RUNG_TAG_PREFIX = "r"


@dataclass(frozen=True, slots=True)
class WantedRung:
    """An entry rung the strategy wants resting at one of its bands."""

    band_index: int
    order: BatchableOrderAction


def entered_side(order_side: OrderSide) -> PositionSide:
    return PositionSide.LONG if order_side is OrderSide.BUY else PositionSide.SHORT


def rung_band_index(client_order_id: str | None) -> int | None:
    """Return the band a resting rung was placed for, or None if it is not ours."""
    tag = tag_of(client_order_id)
    if tag is None or not tag.startswith(_RUNG_TAG_PREFIX):
        return None
    band_index = tag[len(_RUNG_TAG_PREFIX) :]
    return int(band_index) if band_index.isdigit() else None


def wanted_while_flat(
    account: FuturesAccount,
    profile: ProfileConfig,
    profile_snapshot: ProfileSnapshot,
    reentry_state: ReentryState,
    account_snapshot: AccountSnapshot,
    entry_order: EntryOrder,
) -> list[WantedRung]:
    """Return a rung for every band the profile may enter on, both sides."""
    wanted: list[WantedRung] = []
    for side, band_prices in profile_snapshot.bands.items():
        if not reentry_state.is_entry_allowed(side):
            continue
        for band_index, band_price in enumerate(band_prices, start=1):
            wanted.append(
                _build_rung(
                    account,
                    profile,
                    side,
                    band_index,
                    band_price,
                    profile_snapshot.reference,
                    account_snapshot,
                    entry_order,
                )
            )
    return wanted


def wanted_while_holding(
    account: FuturesAccount,
    profile: ProfileConfig,
    profile_snapshot: ProfileSnapshot,
    position: PositionSnapshot,
    reentry_state: ReentryState,
    account_snapshot: AccountSnapshot,
    entry_order: EntryOrder,
) -> list[WantedRung]:
    """Return a rung for each band beyond those the position already filled."""
    if not reentry_state.is_entry_allowed(position.side):
        return []

    bands = profile_snapshot.bands[position.side]
    unfilled_count = sum(
        1
        for rung in profile_snapshot.pending_rungs
        if entered_side(rung.side) is position.side
    )
    first_unfilled_band_index = len(bands) - unfilled_count

    wanted: list[WantedRung] = []
    for band_index, band_price in enumerate(
        bands[first_unfilled_band_index:], start=first_unfilled_band_index + 1
    ):
        wanted.append(
            _build_rung(
                account,
                profile,
                position.side,
                band_index,
                band_price,
                profile_snapshot.reference,
                account_snapshot,
                entry_order,
            )
        )
    return wanted


def _build_rung(
    account: FuturesAccount,
    profile: ProfileConfig,
    side: PositionSide,
    band_index: int,
    band_price: float,
    reference: float,
    account_snapshot: AccountSnapshot,
    entry_order: EntryOrder,
) -> WantedRung:
    entry = (
        account.long_entry(profile.symbol)
        if side is PositionSide.LONG
        else account.short_entry(profile.symbol)
    )
    resting_at_band = entry_order.rest_at_band(entry, band_price)
    return WantedRung(
        band_index=band_index,
        order=resting_at_band.size(
            profile.rung_sizing[side], band_price, account_snapshot
        )
        .stop_loss(
            price=stop_loss_trigger_price(side, band_price, profile.stop_loss_pct)
        )
        .take_profit(price=reference)
        .tag(_rung_tag(band_index))
        .build(),
    )


def _rung_tag(band_index: int) -> str:
    return f"{_RUNG_TAG_PREFIX}{band_index}"
