from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from typing import Any

from robottraderslab import Symbol
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.strategies import (
    AccountSnapshot,
    PositionSide,
    Profile,
    TrackingId,
)
from robottraderslab.strategies.futures import (
    MarginMode,
    SizingRule,
)

BAND_LABEL: dict[PositionSide, str] = {
    PositionSide.LONG: "low",
    PositionSide.SHORT: "high",
}
_ENVELOPE_KEYS = ("envelopes", "long_envelopes", "short_envelopes")


class AverageType(StrEnum):
    """The kinds of average the envelope bands can sit around."""

    SMA = "SMA"
    EMA = "EMA"
    WMA = "WMA"
    DCM = "DCM"


@dataclass(frozen=True, kw_only=True)
class ProfileConfig(Profile):
    """One symbol traded on one timeframe, with its own bands and sizing.

    A symbol names the profile, so the strategy accepts only one profile per
    symbol. `envelopes` gives both sides the same offsets; `long_envelopes`
    and `short_envelopes` give a side its own, alone or together.
    """

    average_type: AverageType
    average_period: int
    stop_loss_pct: float
    sizing: SizingRule
    envelopes: list[float] | None = None
    long_envelopes: list[float] | None = None
    short_envelopes: list[float] | None = None
    leverage: float = 1.0
    margin_mode: MarginMode = MarginMode.ISOLATED

    def __post_init__(self) -> None:
        """
        Raises:
            StrategyCriticalError: If a value is out of range, if `envelopes`
                is combined with a directional list, or if no side has offsets.
        """
        self._validate_envelopes()
        self._validate_leverage()
        self._validate_stop_loss_pct()

    @cached_property
    def band_columns(self) -> dict[PositionSide, tuple[str, ...]]:
        return {
            side: tuple(
                f"{self.profile_id}_band_{BAND_LABEL[side]}_{index}"
                for index in range(1, len(offsets) + 1)
            )
            for side, offsets in self.offsets_by_side.items()
        }

    @cached_property
    def offsets_by_side(self) -> dict[PositionSide, list[float]]:
        """Each side's band offsets; a side the profile does not trade holds none."""
        written = {
            key: getattr(self, key)
            for key in _ENVELOPE_KEYS
            if getattr(self, key) is not None
        }
        return envelopes_by_side(written)

    @cached_property
    def profile_id(self) -> TrackingId:
        return TrackingId(str(self.symbol))

    @cached_property
    def reference_column(self) -> str:
        return f"{self.profile_id}_reference"

    @cached_property
    def rung_counts(self) -> dict[PositionSide, int]:
        """Each side splits the whole quantity the profile's rule sizes over
        its rungs.
        """
        return {
            side: len(offsets)
            for side, offsets in self.offsets_by_side.items()
            if offsets
        }

    @cached_property
    def rung_sizing(self) -> dict[PositionSide, SizingRule]:
        return {
            side: RungShare(self.sizing, rungs)
            for side, rungs in self.rung_counts.items()
        }

    def _validate_envelopes(self) -> None:
        if not any(self.offsets_by_side.values()):
            raise StrategyCriticalError(
                "profile must declare envelopes, long_envelopes or short_envelopes"
            )
        for side, offsets in self.offsets_by_side.items():
            if not all(0 < offset < 1 for offset in offsets):
                raise StrategyCriticalError(
                    f"envelope offsets must be in (0, 1), got {offsets} on the "
                    f"{side.value} side"
                )

    def _validate_leverage(self) -> None:
        if self.leverage <= 0:
            raise StrategyCriticalError(
                f"leverage must be greater than 0, got {self.leverage} on {self.symbol}"
            )

    def _validate_stop_loss_pct(self) -> None:
        if not 0 < self.stop_loss_pct < 1:
            raise StrategyCriticalError(
                f"stop_loss_pct must be in (0, 1), got {self.stop_loss_pct} on "
                f"{self.symbol}"
            )


@dataclass(frozen=True, slots=True)
class RungShare(SizingRule):
    """A rung trades an equal share of what the profile's rule sizes for its
    side.
    """

    rule: SizingRule
    rungs: int

    def quantity(
        self,
        symbol: Symbol,
        price: float,
        account_snapshot: AccountSnapshot,
        *,
        placement_reserve_rate: float,
        placement_requirement_rate: Callable[[float], float],
        stop_loss_price: float | None,
    ) -> float:
        return (
            self.rule.quantity(
                symbol,
                price,
                account_snapshot,
                placement_reserve_rate=placement_reserve_rate,
                placement_requirement_rate=placement_requirement_rate,
                stop_loss_price=stop_loss_price,
            )
            / self.rungs
        )


def envelopes_by_side(settings: Mapping[str, Any]) -> dict[PositionSide, list[float]]:
    """Read each side's offsets from the envelope keys a profile spells.

    A side no key names carries an empty list, so a caller reads either side
    without checking whether the profile trades it; a key that is written holds
    at least one offset, so an emptied list is a mistake and not a silent opt-out.

    Raises:
        StrategyCriticalError: If `envelopes` is combined with `long_envelopes`
            or `short_envelopes`, or if a written key holds an empty list.
    """
    for key in _ENVELOPE_KEYS:
        if key in settings and not settings[key]:
            raise StrategyCriticalError(f"{key} must be a non-empty list")
    both_sides = settings.get("envelopes")
    long_offsets = settings.get("long_envelopes")
    short_offsets = settings.get("short_envelopes")
    if both_sides is None:
        return {
            PositionSide.LONG: long_offsets or [],
            PositionSide.SHORT: short_offsets or [],
        }
    if long_offsets is not None or short_offsets is not None:
        raise StrategyCriticalError(
            "envelopes cannot be combined with long_envelopes or short_envelopes"
        )
    return {PositionSide.LONG: both_sides, PositionSide.SHORT: both_sides}
