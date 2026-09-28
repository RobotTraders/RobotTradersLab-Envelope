from robottraderslab.strategies import BookKeeper, PositionSide
from robottraderslab.strategies.futures import FuturesAccount, PositionSnapshot

from .profile_config import ProfileConfig
from .profile_snapshot import ProfileSnapshot


def book_position_protections(
    account: FuturesAccount,
    profile: ProfileConfig,
    profile_snapshot: ProfileSnapshot,
    position: PositionSnapshot,
    bookkeeper: BookKeeper,
) -> None:
    """Move the position's stop-loss and take-profit to this candle's levels."""
    bookkeeper.add(
        account.move_stop_loss(
            position.symbol,
            stop_loss_trigger_price(
                position.side, position.average_entry_price, profile.stop_loss_pct
            ),
        ),
        after=(),
    )
    bookkeeper.add(
        account.move_take_profit(position.symbol, profile_snapshot.reference),
        after=(),
    )


def stop_loss_trigger_price(
    side: PositionSide, reference_price: float, stop_loss_pct: float
) -> float:
    """Return the trigger price a stop-loss sits at, on either side."""
    if side is PositionSide.LONG:
        return reference_price * (1 - stop_loss_pct)
    return reference_price * (1 + stop_loss_pct)
