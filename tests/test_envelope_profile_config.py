import re

import pytest

from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.strategies import PositionSide
from robottraderslab.strategies.futures import (
    MarginMode,
    TotalBalanceRatio,
)
from robottraderslab_envelope.profile_config import AverageType, ProfileConfig

SYMBOL = "BTC/USDT:USDT"
LONG = PositionSide.LONG
SHORT = PositionSide.SHORT


def _kwargs(**overrides: object) -> dict[str, object]:
    defaults: dict[str, object] = {
        "symbol": SYMBOL,
        "timeframe": "1h",
        "sizing": TotalBalanceRatio(1.0),
        "average_type": AverageType.SMA,
        "average_period": 100,
        "stop_loss_pct": 0.15,
    }
    defaults.update(overrides)
    return defaults


def _section(**overrides: object) -> dict[str, object]:
    section: dict[str, object] = {
        "symbol": SYMBOL,
        "timeframe": "1h",
        "total_balance_ratio": 1.0,
        "average_type": "SMA",
        "average_period": 100,
        "envelopes": [0.05],
        "stop_loss_pct": 0.15,
    }
    section.update(overrides)
    return section


class TestConstruction:
    def test_derives_profile_id_from_symbol(self):
        profile = ProfileConfig(**_kwargs(envelopes=[0.07, 0.11, 0.14]))

        assert str(profile.profile_id) == "BTC/USDT:USDT"

    def test_defaults_applied(self):
        profile = ProfileConfig(**_kwargs(envelopes=[0.07, 0.11, 0.14]))

        assert profile.leverage == 1.0
        assert profile.margin_mode is MarginMode.ISOLATED


class TestBuiltFromASection:
    def test_the_strings_a_section_spells(self, make_strategy):
        strategy = make_strategy([_section(margin_mode="cross", average_type="DCM")])

        (profile,) = strategy.profiles
        assert profile.margin_mode is MarginMode.CROSS
        assert profile.average_type is AverageType.DCM

    def test_unknown_margin_mode_rejected(self, make_strategy):
        with pytest.raises(StrategyCriticalError, match="margin_mode: Input should be"):
            make_strategy([_section(margin_mode="nonsense")])

    def test_unknown_average_type_rejected(self, make_strategy):
        with pytest.raises(
            StrategyCriticalError, match="average_type: Input should be"
        ):
            make_strategy([_section(average_type="DCA")])


class TestLevelColumns:
    def test_reference_column_named_after_profile(self):
        profile = ProfileConfig(**_kwargs(envelopes=[0.07, 0.11, 0.14]))

        assert profile.reference_column == "BTC/USDT:USDT_reference"

    def test_band_columns_named_low_and_high_per_index(self):
        profile = ProfileConfig(**_kwargs(envelopes=[0.05, 0.10]))

        assert profile.band_columns == {
            LONG: ("BTC/USDT:USDT_band_low_1", "BTC/USDT:USDT_band_low_2"),
            SHORT: ("BTC/USDT:USDT_band_high_1", "BTC/USDT:USDT_band_high_2"),
        }

    def test_band_columns_reflect_each_sides_own_count(self):
        profile = ProfileConfig(
            **_kwargs(long_envelopes=[0.05, 0.10, 0.15], short_envelopes=[0.05]),
        )

        assert len(profile.band_columns[LONG]) == 3
        assert len(profile.band_columns[SHORT]) == 1


class TestEnvelopesValidation:
    def test_one_side_empty_is_allowed(self):
        profile = ProfileConfig(**_kwargs(long_envelopes=[0.05]))

        assert profile.offsets_by_side == {LONG: [0.05], SHORT: []}

    @pytest.mark.parametrize(
        "offsets", [[0.0, 0.1], [-0.05, 0.1], [0.1, 1.0], [0.1, 1.2]]
    )
    def test_offset_outside_the_unit_interval_rejected(self, offsets):
        with pytest.raises(StrategyCriticalError, match="must be in"):
            ProfileConfig(**_kwargs(long_envelopes=offsets, short_envelopes=[0.1]))


class TestLeverageValidation:
    @pytest.mark.parametrize("leverage", [0.0, -1.0])
    def test_not_above_zero_rejected(self, leverage):
        with pytest.raises(
            StrategyCriticalError, match=rf"leverage.*{re.escape(SYMBOL)}"
        ):
            ProfileConfig(**_kwargs(envelopes=[0.05], leverage=leverage))


class TestStopLossPctValidation:
    @pytest.mark.parametrize("stop_loss_pct", [0.0, -0.1, 1.0, 1.5])
    def test_outside_the_unit_interval_rejected(self, stop_loss_pct):
        with pytest.raises(
            StrategyCriticalError, match=rf"stop_loss_pct.*{re.escape(SYMBOL)}"
        ):
            ProfileConfig(**_kwargs(envelopes=[0.05], stop_loss_pct=stop_loss_pct))


class TestEnvelopeForms:
    def test_envelopes_applies_to_both_sides(self):
        profile = ProfileConfig(**_kwargs(envelopes=[0.05, 0.10]))

        assert profile.offsets_by_side == {LONG: [0.05, 0.10], SHORT: [0.05, 0.10]}

    def test_long_envelopes_alone_leaves_short_side_empty(self):
        profile = ProfileConfig(**_kwargs(long_envelopes=[0.05, 0.10]))

        assert profile.offsets_by_side == {LONG: [0.05, 0.10], SHORT: []}

    def test_short_envelopes_alone_leaves_long_side_empty(self):
        profile = ProfileConfig(**_kwargs(short_envelopes=[0.05, 0.10]))

        assert profile.offsets_by_side == {LONG: [], SHORT: [0.05, 0.10]}

    def test_both_directional_lists_declared_together(self):
        profile = ProfileConfig(
            **_kwargs(long_envelopes=[0.05, 0.10, 0.15], short_envelopes=[0.08]),
        )

        assert profile.offsets_by_side == {LONG: [0.05, 0.10, 0.15], SHORT: [0.08]}

    @pytest.mark.parametrize(
        "directional", [{"long_envelopes": [0.05]}, {"short_envelopes": [0.05]}]
    )
    def test_envelopes_with_a_directional_list_rejected(self, directional):
        with pytest.raises(StrategyCriticalError, match="cannot be combined"):
            ProfileConfig(**_kwargs(envelopes=[0.05], **directional))

    def test_no_form_declared_rejected(self):
        with pytest.raises(StrategyCriticalError, match="must declare"):
            ProfileConfig(**_kwargs())

    @pytest.mark.parametrize("key", ["envelopes", "long_envelopes", "short_envelopes"])
    def test_a_written_key_with_an_empty_list_rejected(self, key):
        with pytest.raises(StrategyCriticalError, match=f"{key} must be a non-empty"):
            ProfileConfig(**_kwargs(**{key: []}))

    def test_offset_out_of_range_rejected(self):
        with pytest.raises(StrategyCriticalError, match="must be in"):
            ProfileConfig(**_kwargs(long_envelopes=[1.2]))


class TestRungCounts:
    def test_each_traded_side_counts_its_own_rungs(self):
        profile = ProfileConfig(
            **_kwargs(long_envelopes=[0.05, 0.10, 0.15], short_envelopes=[0.05])
        )

        assert profile.rung_counts == {LONG: 3, SHORT: 1}

    def test_a_side_without_offsets_counts_none(self):
        profile = ProfileConfig(**_kwargs(long_envelopes=[0.05, 0.10]))

        assert profile.rung_counts == {LONG: 2}
