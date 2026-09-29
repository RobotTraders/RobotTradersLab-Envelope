import asyncio
import logging
from collections.abc import Callable
from typing import Any

import pytest

from robottraderslab import Symbol
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.strategies import StrategyRequirements
from robottraderslab_envelope import EnvelopeStrategy


def _sized_profile(
    symbol: str, total_balance_ratio: float, leverage: float
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "timeframe": "1d",
        "average_type": "SMA",
        "average_period": 100,
        "envelopes": [0.05, 0.10],
        "stop_loss_pct": 0.04,
        "total_balance_ratio": total_balance_ratio,
        "leverage": leverage,
    }


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]


class TestSetup:
    @pytest.fixture
    def single_profile_strategy(
        self, make_strategy: Callable[..., EnvelopeStrategy]
    ) -> EnvelopeStrategy:
        return make_strategy(
            [
                {
                    "symbol": "BTC/USDT:USDT",
                    "timeframe": "1d",
                    "total_balance_ratio": 1.0,
                    "average_type": "SMA",
                    "average_period": 100,
                    "envelopes": [0.05, 0.10, 0.15],
                    "stop_loss_pct": 0.04,
                }
            ]
        )

    def test_registers_requirement_per_profile(self, single_profile_strategy):
        requirements = StrategyRequirements()

        asyncio.run(single_profile_strategy.setup(requirements))

        registered = requirements.ohlcv._get_all()
        assert len(registered) == 1
        assert registered[0].symbol == Symbol.create("BTC/USDT:USDT")
        assert registered[0].timeframe == "1d"
        assert registered[0].lookback == 100

    def test_multiple_profiles_register_each_requirement(self, make_strategy):
        strategy = make_strategy(
            [
                {
                    "symbol": "BTC/USDT:USDT",
                    "timeframe": "1d",
                    "total_balance_ratio": 1.0,
                    "average_type": "SMA",
                    "average_period": 100,
                    "envelopes": [0.05],
                    "stop_loss_pct": 0.04,
                },
                {
                    "symbol": "ETH/USDT:USDT",
                    "timeframe": "1h",
                    "total_balance_ratio": 1.0,
                    "average_type": "EMA",
                    "average_period": 50,
                    "envelopes": [0.03, 0.06],
                    "stop_loss_pct": 0.02,
                },
            ]
        )
        requirements = StrategyRequirements()

        asyncio.run(strategy.setup(requirements))

        registered = requirements.ohlcv._get_all()
        assert len(registered) == 2

    def test_duplicate_symbol_across_profiles_rejected(self, make_strategy):
        strategy = make_strategy(
            [
                {
                    "symbol": "BTC/USDT:USDT",
                    "timeframe": "1d",
                    "total_balance_ratio": 1.0,
                    "average_type": "SMA",
                    "average_period": 100,
                    "envelopes": [0.05],
                    "stop_loss_pct": 0.04,
                },
                {
                    "symbol": "BTC/USDT:USDT",
                    "timeframe": "4h",
                    "total_balance_ratio": 1.0,
                    "average_type": "EMA",
                    "average_period": 50,
                    "envelopes": [0.03],
                    "stop_loss_pct": 0.02,
                },
            ]
        )
        requirements = StrategyRequirements()

        with pytest.raises(StrategyCriticalError, match="BTC/USDT:USDT"):
            asyncio.run(strategy.setup(requirements))

    def test_unknown_update_mode(self, make_strategy):
        profile = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1d",
            "total_balance_ratio": 1.0,
            "average_type": "SMA",
            "average_period": 100,
            "envelopes": [0.05],
            "stop_loss_pct": 0.04,
        }

        with pytest.raises(StrategyCriticalError, match="update_mode"):
            make_strategy([profile], update_mode="amend")

    def test_unknown_entry_order(self, make_strategy):
        profile = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1d",
            "total_balance_ratio": 1.0,
            "average_type": "SMA",
            "average_period": 100,
            "envelopes": [0.05],
            "stop_loss_pct": 0.04,
        }

        with pytest.raises(StrategyCriticalError, match="'trigger' or 'limit'"):
            make_strategy([profile], entry_order="stop")


class TestRestingRungBalance:
    def test_rungs_outgrowing_the_balance(self, make_strategy, caplog):
        strategy = make_strategy(
            [
                _sized_profile("BTC/USDT:USDT", total_balance_ratio=0.6, leverage=2.0),
                _sized_profile("ETH/USDT:USDT", total_balance_ratio=0.6, leverage=2.0),
            ]
        )

        with caplog.at_level(logging.WARNING):
            asyncio.run(strategy.setup(StrategyRequirements()))

        assert _warnings(caplog) == [
            "The resting rungs of every profile lock 120% of the balance; "
            "the venue refuses the rungs beyond it."
        ]

    def test_rungs_fitting_the_balance(self, make_strategy, caplog):
        strategy = make_strategy(
            [
                _sized_profile("BTC/USDT:USDT", total_balance_ratio=0.4, leverage=2.0),
                _sized_profile("ETH/USDT:USDT", total_balance_ratio=0.4, leverage=2.0),
            ]
        )

        with caplog.at_level(logging.WARNING):
            asyncio.run(strategy.setup(StrategyRequirements()))

        assert _warnings(caplog) == []

    def test_a_side_the_profile_leaves_out_rests_nothing(self, make_strategy, caplog):
        profile = _sized_profile("BTC/USDT:USDT", total_balance_ratio=1.0, leverage=1.0)
        profile["long_envelopes"] = profile.pop("envelopes")
        strategy = make_strategy([profile])

        with caplog.at_level(logging.WARNING):
            asyncio.run(strategy.setup(StrategyRequirements()))

        assert _warnings(caplog) == []


class TestSizingReads:
    def test_a_rule_reading_equity_declares_it(self, make_strategy):
        profile = _sized_profile("BTC/USDT:USDT", total_balance_ratio=0.5, leverage=1.0)
        profile["equity_ratio"] = profile.pop("total_balance_ratio")
        strategy = make_strategy([profile])
        requirements = StrategyRequirements()

        asyncio.run(strategy.setup(requirements))

        (account_requirement,) = requirements.account._get_all()
        assert account_requirement.equity == ("USDT",)

    def test_a_rule_reading_the_balance_declares_no_equity(self, make_strategy):
        strategy = make_strategy(
            [_sized_profile("BTC/USDT:USDT", total_balance_ratio=0.5, leverage=1.0)]
        )
        requirements = StrategyRequirements()

        asyncio.run(strategy.setup(requirements))

        (account_requirement,) = requirements.account._get_all()
        assert account_requirement.equity == ()

    def test_a_rule_sizing_from_no_share_of_the_balance_warns_of_nothing(
        self, make_strategy, caplog
    ):
        profile = _sized_profile("BTC/USDT:USDT", total_balance_ratio=1.0, leverage=0.5)
        profile["notional"] = 1_000_000.0
        del profile["total_balance_ratio"]
        strategy = make_strategy([profile])

        with caplog.at_level(logging.WARNING):
            asyncio.run(strategy.setup(StrategyRequirements()))

        assert _warnings(caplog) == []
