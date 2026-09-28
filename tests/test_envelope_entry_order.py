from unittest.mock import Mock

import pytest
from robottraderslab_envelope.entry_order import LimitEntryOrder, TriggerEntryOrder

from robottraderslab.strategies.futures import FuturesOrderBuilder


@pytest.fixture
def entry() -> Mock:
    return Mock(spec=FuturesOrderBuilder)


class TestRestAtBand:
    def test_limit_rests_at_the_band_price(self, entry):
        resting = LimitEntryOrder().rest_at_band(entry, 95.0)

        entry.limit.assert_called_once_with(price=95.0)
        entry.trigger.assert_not_called()
        assert resting is entry.limit.return_value

    def test_trigger_rests_at_the_band_price(self, entry):
        resting = TriggerEntryOrder().rest_at_band(entry, 95.0)

        entry.trigger.assert_called_once_with(price=95.0)
        entry.limit.assert_not_called()
        assert resting is entry.trigger.return_value


class TestMatches:
    @pytest.mark.parametrize(
        ("kind", "order_kind", "expected"),
        [
            (LimitEntryOrder(), "limit", True),
            (LimitEntryOrder(), "trigger", False),
            (TriggerEntryOrder(), "trigger", True),
            (TriggerEntryOrder(), "limit", False),
            (LimitEntryOrder(), None, False),
            (TriggerEntryOrder(), None, False),
        ],
    )
    def test_matches_the_configured_kind_alone(self, kind, order_kind, expected):
        assert kind.matches(order_kind) is expected
