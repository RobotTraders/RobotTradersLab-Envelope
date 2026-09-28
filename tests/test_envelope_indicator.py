import pandas as pd
import pytest
from robottraderslab_envelope.envelope_indicator import compute_envelope
from robottraderslab_envelope.profile_config import AverageType

from robottraderslab.indicators import MAType
from robottraderslab.strategies import PositionSide

LONG = PositionSide.LONG
SHORT = PositionSide.SHORT


class TestComputeEnvelope:
    def test_returns_one_band_per_offset_in_configured_order(self):
        reference = pd.Series([100.0, 101.0, 102.0])

        levels = compute_envelope(reference, {LONG: [0.05, 0.10], SHORT: [0.05]})

        assert len(levels.bands[LONG]) == 2
        assert levels.bands[LONG][0].iloc[0] == pytest.approx(100.0 * 0.95)
        assert levels.bands[LONG][1].iloc[0] == pytest.approx(100.0 * 0.90)
        assert len(levels.bands[SHORT]) == 1

    def test_a_fall_of_the_offset_lands_on_the_reference_from_either_band(self):
        reference = pd.Series([100.0])

        levels = compute_envelope(reference, {LONG: [0.10], SHORT: [0.10]})

        band_low = levels.bands[LONG][0].iloc[0]
        band_high = levels.bands[SHORT][0].iloc[0]
        assert band_low == pytest.approx(90.0)
        assert band_high == pytest.approx(111.111111)
        assert band_high * (1 - 0.10) == pytest.approx(reference.iloc[0])

    def test_preserves_reference_series(self):
        reference = pd.Series([100.0, 105.0, 110.0])

        levels = compute_envelope(reference, {LONG: [0.05], SHORT: [0.05]})

        pd.testing.assert_series_equal(levels.reference, reference, check_names=False)

    def test_preserves_index(self):
        index = pd.date_range("2024-01-01", periods=3, freq="h")
        reference = pd.Series([100.0, 101.0, 102.0], index=index)

        levels = compute_envelope(reference, {LONG: [0.05], SHORT: [0.05]})

        assert (levels.bands[LONG][0].index == index).all()
        assert (levels.bands[SHORT][0].index == index).all()


# compute_reference casts an AverageType back into an MAType by value, so a
# moving average the framework gains must be offered by the profile too.
@pytest.mark.parametrize("moving_average_type", list(MAType))
def test_every_moving_average_type_is_an_average_type(moving_average_type):
    assert moving_average_type.value in {average.value for average in AverageType}
