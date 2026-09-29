import numpy as np
import pytest

from robottraderslab.strategies import Candles, ChartLine
from robottraderslab_envelope.chart_indicators import get_lightweight_chart_indicators

CLOSES = np.array([100.0 + i * 2.0 for i in range(50)])
CANDLES = Candles(
    open=CLOSES,
    high=CLOSES + 8.0,
    low=CLOSES - 2.0,
    close=CLOSES,
    volume=np.ones_like(CLOSES),
)
AVERAGE_PERIOD = 5
PARAMS = {
    "average_type": "SMA",
    "average_period": AVERAGE_PERIOD,
    "envelopes": [0.04, 0.06],
}
SETTLED = slice(AVERAGE_PERIOD, None)


def _named(lines: list[ChartLine]) -> dict[str, ChartLine]:
    return {line.name: line for line in lines}


class TestGetLightweightChartIndicators:
    @pytest.fixture
    def lines(self) -> dict[str, ChartLine]:
        return _named(get_lightweight_chart_indicators(CANDLES, PARAMS))

    def test_names_the_reference_after_its_average(self, lines):
        assert "Reference SMA 5" in lines

    def test_draws_a_low_and_a_high_line_per_offset(self, lines):
        assert set(lines) == {
            "Reference SMA 5",
            "Band 1 low",
            "Band 1 high",
            "Band 2 low",
            "Band 2 high",
        }

    def test_a_band_brackets_the_reference(self, lines):
        reference = lines["Reference SMA 5"].values[SETTLED]

        assert (lines["Band 1 low"].values[SETTLED] < reference).all()
        assert (lines["Band 1 high"].values[SETTLED] > reference).all()

    def test_the_wider_offset_sits_further_out(self, lines):
        wider = lines["Band 2 low"].values[SETTLED]

        assert (wider < lines["Band 1 low"].values[SETTLED]).all()

    def test_a_donchian_reference_reads_the_highs_and_lows(self, lines):
        donchian = _named(
            get_lightweight_chart_indicators(CANDLES, {**PARAMS, "average_type": "DCM"})
        )

        assert not np.allclose(
            lines["Reference SMA 5"].values,
            donchian["Reference DCM 5"].values,
            equal_nan=True,
        )

    def test_a_line_carries_one_value_per_candle(self, lines):
        assert all(len(line.values) == len(CLOSES) for line in lines.values())


class TestDirectionalForms:
    def test_long_envelopes_alone_draws_only_low_bands(self):
        lines = _named(
            get_lightweight_chart_indicators(
                CANDLES,
                {
                    "average_type": "SMA",
                    "average_period": AVERAGE_PERIOD,
                    "long_envelopes": [0.04, 0.06],
                },
            )
        )

        assert set(lines) == {"Reference SMA 5", "Band 1 low", "Band 2 low"}

    def test_short_envelopes_alone_draws_only_high_bands(self):
        lines = _named(
            get_lightweight_chart_indicators(
                CANDLES,
                {
                    "average_type": "SMA",
                    "average_period": AVERAGE_PERIOD,
                    "short_envelopes": [0.04],
                },
            )
        )

        assert set(lines) == {"Reference SMA 5", "Band 1 high"}
