from typing import Any

from robottraderslab.strategies import Candles, ChartLine, PositionSide

from .envelope_indicator import compute_envelope, compute_reference
from .profile_config import BAND_LABEL, AverageType, envelopes_by_side

_BAND_COLOUR = {PositionSide.LONG: "green", PositionSide.SHORT: "red"}


def get_lightweight_chart_indicators(
    candles: Candles,
    indicator_params: dict[str, Any],
) -> list[ChartLine]:
    """Draw the reference average and the band a rung rests on beside the candles.

    Args:
        indicator_params: `average_type`, `average_period` and either
            `envelopes` or `long_envelopes` / `short_envelopes`, spelled as a
            profile spells them.
    """
    average_type = AverageType(indicator_params["average_type"])
    average_period = indicator_params["average_period"]
    reference = compute_reference(
        candles.close,
        candles.high,
        candles.low,
        average_type,
        average_period,
    )
    levels = compute_envelope(reference, envelopes_by_side(indicator_params))

    lines = [
        ChartLine(
            name=f"Reference {average_type} {average_period}",
            values=levels.reference,
            colour="orange",
        )
    ]
    for side, bands in levels.bands.items():
        for index, band in enumerate(bands, start=1):
            lines.append(
                ChartLine(
                    name=f"Band {index} {BAND_LABEL[side]}",
                    values=band,
                    colour=_BAND_COLOUR[side],
                )
            )
    return lines
