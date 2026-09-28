import logging
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from typing import ClassVar, Literal

from robottraderslab import Symbol
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.strategies import (
    AccountSnapshots,
    BookKeeper,
    Execution,
    MarketType,
    OHLCVs,
    OrderType,
    ProfileStrategy,
    StrategyRequirements,
)
from robottraderslab.strategies.futures import (
    FuturesAccount,
    MarginSettings,
    TotalBalanceRatio,
)

from .book_rungs import (
    RungUpdater,
    book_cancel_and_place,
    book_cancel_entries,
    book_modify_in_place,
)
from .entry_order import EntryOrder, LimitEntryOrder, TriggerEntryOrder
from .envelope_indicator import compute_envelope, compute_reference
from .position_protection import book_position_protections
from .profile_config import ProfileConfig
from .profile_snapshot import ProfileSnapshot
from .reentry_guard import ReentryGuard
from .rungs import (
    entered_side,
    rung_band_index,
    wanted_while_flat,
    wanted_while_holding,
)

logger = logging.getLogger(__name__)

_REFERENCE_EXIT_REASON = "moving average reached"
_ENTRY_ORDERS: dict[str, EntryOrder] = {
    "limit": LimitEntryOrder(),
    "trigger": TriggerEntryOrder(),
}
_RUNG_UPDATERS: dict[str, RungUpdater] = {
    "modify": book_modify_in_place,
    "cancel_replace": book_cancel_and_place,
}
_WHOLE_BALANCE = 1.0


class EnvelopeStrategy(ProfileStrategy[ProfileConfig]):
    """Mean-reversion envelope strategy: rests one entry rung per band around a
    moving average.

    The `entry_order` setting chooses what a rung rests as: "trigger" fires a
    market order once price reaches the band, "limit" rests at the band price
    and fills there or better.

    The `update_mode` setting chooses how the resting rungs follow the bands:
    "modify" moves each resting rung to its band so the book is never empty,
    "cancel_replace" cancels the resting set and places the wanted one afresh.
    """

    market_type: ClassVar[MarketType] = "futures"

    account: FuturesAccount
    entry_order: Literal["trigger", "limit"] = "trigger"
    update_mode: Literal["modify", "cancel_replace"] = "modify"

    _entry_order: EntryOrder
    _profiles_by_symbol: dict[Symbol, ProfileConfig]
    _reentry_guard: ReentryGuard
    _update_rungs: RungUpdater

    async def setup(self, requirements: StrategyRequirements) -> None:
        """Resolve the settings, validate the profiles and declare the OHLCV and
        account state requirements.
        """
        self._entry_order = _ENTRY_ORDERS[self.entry_order]
        self._update_rungs = _RUNG_UPDATERS[self.update_mode]
        self._reentry_guard = ReentryGuard()
        self._profiles_by_symbol = {
            profile.symbol: profile for profile in self.profiles
        }
        _validate_profiles(self.profiles)
        if self.account.capabilities.reserves_margin_on_pending_orders:
            logger.info(
                "The exchange reserves margin on pending orders; the envelope "
                "keeps many resting orders open at once, so only the rungs "
                "covered by free margin will fill."
            )
        _warn_when_resting_rungs_outgrow_the_balance(self.profiles)
        for profile in self.profiles:
            requirements.ohlcv.add(
                profile.symbol, profile.timeframe, profile.average_period
            )
        requirements.account.add(
            self.account,
            symbols=[profile.symbol for profile in self.profiles],
            positions=True,
            open_orders=True,
            balances=True,
            sizing=[profile.sizing for profile in self.profiles],
            executions=True,
            notify_entry_fills=True,
            describe_fills=self._describe_fill,
            margin_targets=_margin_targets(self.profiles),
        )

    def generate_profile_signals(self, profile: ProfileConfig, ohlcvs: OHLCVs) -> None:
        """Compute the profile's reference moving average and envelope bands."""
        symbol = profile.symbol
        timeframe = profile.timeframe
        reference = compute_reference(
            ohlcvs.column(symbol, timeframe, "close"),
            ohlcvs.column(symbol, timeframe, "high"),
            ohlcvs.column(symbol, timeframe, "low"),
            profile.average_type,
            profile.average_period,
        )
        levels = compute_envelope(reference, profile.offsets_by_side)
        ohlcvs.add_column(symbol, timeframe, profile.reference_column, levels.reference)
        for side, columns in profile.band_columns.items():
            for column, band in zip(columns, levels.bands[side], strict=True):
                ohlcvs.add_column(symbol, timeframe, column, band)

    def book_profile_actions(
        self,
        profile: ProfileConfig,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timestamp: datetime,
        bookkeeper: BookKeeper,
    ) -> None:
        """Emit the profile's rung-update, entry and protection actions."""
        account_snapshot = account_snapshots.of(self.account)
        profile_snapshot = ProfileSnapshot.build(
            profile, ohlcvs, account_snapshot, self._entry_order
        )
        if profile_snapshot.has_nan_levels:
            logger.warning(
                f"{profile.profile_id} ({profile.symbol}): envelope levels are NaN, "
                "cancelling entries and waiting for data to recover"
            )
            book_cancel_entries(self.account, profile, profile_snapshot, bookkeeper)
            return

        reentry_state = self._reentry_guard.state_for(
            profile.symbol,
            profile.timeframe,
            profile.reference_column,
            ohlcvs,
            account_snapshot.stop_loss_fills(profile.symbol),
            timestamp,
        )
        position = profile_snapshot.position
        if position is not None:
            book_position_protections(
                self.account,
                profile,
                profile_snapshot,
                position,
                bookkeeper,
            )
            wanted = wanted_while_holding(
                self.account,
                profile,
                profile_snapshot,
                position,
                reentry_state,
                account_snapshot,
                self._entry_order,
            )
        else:
            wanted = wanted_while_flat(
                self.account,
                profile,
                profile_snapshot,
                reentry_state,
                account_snapshot,
                self._entry_order,
            )

        self._update_rungs(
            self.account,
            profile,
            profile_snapshot,
            wanted,
            bookkeeper,
        )

    def _describe_fill(self, fill: Execution) -> str | None:
        """A take-profit sits at the reference, so its fill is the envelope's exit.

        The venue reports every fill on the symbol, so a fill of the configured
        entry kind is one of the envelope's rungs only when the tag on its
        client order id says so.
        """
        if fill.kind == OrderType.TAKE_PROFIT:
            return _REFERENCE_EXIT_REASON
        if not self._entry_order.matches(fill.kind):
            return None
        band_index = rung_band_index(fill.client_order_id)
        if band_index is None:
            return None
        offsets = self._profiles_by_symbol[fill.symbol].offsets_by_side
        return f"entry rung {band_index} of {len(offsets[entered_side(fill.side)])}"


def _validate_profiles(profiles: Sequence[ProfileConfig]) -> None:
    counts = Counter(profile.symbol for profile in profiles)
    duplicated = sorted(str(symbol) for symbol, count in counts.items() if count > 1)
    if duplicated:
        raise StrategyCriticalError(
            f"Envelope strategy accepts one profile per symbol, "
            f"got several for: {', '.join(duplicated)}."
        )


def _warn_when_resting_rungs_outgrow_the_balance(
    profiles: Sequence[ProfileConfig],
) -> None:
    share = sum(_resting_balance_share(profile) for profile in profiles)
    if share > _WHOLE_BALANCE:
        logger.warning(
            f"The resting rungs of every profile lock {share:.0%} of the "
            "balance; the venue refuses the rungs beyond it."
        )


def _resting_balance_share(profile: ProfileConfig) -> float:
    """A resting rung locks the margin of what it would open, and only a
    share of the total balance is known before the first candle.
    """
    if not isinstance(profile.sizing, TotalBalanceRatio):
        return 0.0
    traded_sides = len(profile.rung_counts)
    return profile.sizing.ratio * traded_sides / profile.leverage


def _margin_targets(
    profiles: Sequence[ProfileConfig],
) -> dict[Symbol, MarginSettings]:
    return {
        profile.symbol: MarginSettings(
            leverage=profile.leverage, margin_mode=profile.margin_mode
        )
        for profile in profiles
    }
