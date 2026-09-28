from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from robottraderslab.indicators import MAType, donchian_midline, moving_average
from robottraderslab.strategies import PositionSide

from .profile_config import AverageType


@dataclass(frozen=True, slots=True)
class EnvelopeLevels:
    """Each side's bands keep the order of the offsets they were computed from."""

    reference: npt.NDArray[np.float64]
    bands: dict[PositionSide, list[npt.NDArray[np.float64]]]


def compute_envelope(
    reference: npt.NDArray[np.float64],
    envelopes: Mapping[PositionSide, Iterable[float]],
) -> EnvelopeLevels:
    """A band at offset `e` is a fall of `e` in price: from the reference down
    to the long band, and from the short band back down to the reference, so
    the short band sits `e / (1 - e)` above the reference.
    """
    return EnvelopeLevels(
        reference=reference,
        bands={
            side: [_band(reference, side, offset) for offset in offsets]
            for side, offsets in envelopes.items()
        },
    )


def compute_reference(
    close: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    average_type: AverageType,
    period: int,
) -> npt.NDArray[np.float64]:
    """Return the reference the bands sit around, of the kind the profile asks for."""
    if average_type is AverageType.DCM:
        return donchian_midline(high, low, period)
    return moving_average(close, period, MAType(average_type.value))


def _band(
    reference: npt.NDArray[np.float64], side: PositionSide, offset: float
) -> npt.NDArray[np.float64]:
    if side is PositionSide.LONG:
        return reference * (1 - offset)
    return reference / (1 - offset)
