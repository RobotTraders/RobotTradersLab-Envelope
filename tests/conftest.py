import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from robottraderslab.exchanges import FuturesExchangeProtocol, Instrument
from robottraderslab.strategies import Balance, TradingMode, TradingSystem
from robottraderslab.strategies.futures import (
    FuturesAccount,
    FuturesCapabilities,
    MarginMode,
    MarginSettings,
)
from robottraderslab_envelope import EnvelopeStrategy


@pytest.fixture
def mock_exchange() -> Mock:
    """Mocked futures exchange with a flat account and no open positions."""
    exchange = Mock(spec=FuturesExchangeProtocol)
    exchange.placement_reserve_rate = 0.0
    exchange.get_balances.return_value = {
        "USDT": Balance(locked=0.0, total=10_000.0),
    }
    exchange.get_open_positions.return_value = {}
    exchange.get_equity.return_value = 10_000.0
    exchange.get_quote_conversion_rate.return_value = 1.0
    exchange.get_open_orders.return_value = []
    exchange.get_executions_since.return_value = []
    exchange.get_margin_settings.side_effect = lambda symbols: {
        symbol: MarginSettings(leverage=None, margin_mode=MarginMode.CROSS)
        for symbol in symbols
    }
    exchange.get_instrument.return_value = Instrument(
        quantity_step=1e-8,
        minimum_order_value=None,
        maximum_leverage=None,
        maximum_order_quantity=None,
        maximum_market_order_quantity=None,
    )
    exchange.capabilities = FuturesCapabilities()
    return exchange


@pytest.fixture(params=["trigger", "limit"])
def entry_order(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def make_strategy(
    mock_exchange: Mock,
) -> Callable[..., EnvelopeStrategy]:
    """Factory for creating EnvelopeStrategy backed by the mocked exchange."""

    def _make(
        profiles: list[dict[str, Any]],
        trading_mode: TradingMode = TradingMode.BACKTEST,
        **strategy_kwargs: Any,
    ) -> EnvelopeStrategy:
        account = FuturesAccount(exchange=mock_exchange, name="test")
        strategy = EnvelopeStrategy(
            account=account,
            trading_system=TradingSystem(trading_mode=trading_mode),
            config_dir=Path("bot-config-dir"),
            profiles=profiles,
            **strategy_kwargs,
        )
        asyncio.run(
            account._read_instruments({profile.symbol for profile in strategy.profiles})
        )
        return strategy

    return _make
