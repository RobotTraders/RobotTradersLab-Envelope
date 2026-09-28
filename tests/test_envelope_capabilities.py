import asyncio
import logging

from robottraderslab.strategies import StrategyRequirements
from robottraderslab.strategies.futures import FuturesCapabilities

_PROFILE = {
    "symbol": "BTC/USDT:USDT",
    "timeframe": "1d",
    "total_balance_ratio": 1.0,
    "average_type": "SMA",
    "average_period": 100,
    "envelopes": [0.05],
    "stop_loss_pct": 0.04,
}
_NOTICE = "reserves margin on pending orders"


def test_logs_notice_when_exchange_reserves_margin_on_pending_orders(
    make_strategy, mock_exchange, caplog
):
    mock_exchange.capabilities = FuturesCapabilities(
        reserves_margin_on_pending_orders=True
    )
    strategy = make_strategy([_PROFILE])

    with caplog.at_level(logging.INFO):
        asyncio.run(strategy.setup(StrategyRequirements()))

    assert _NOTICE in caplog.text


def test_no_notice_when_exchange_does_not_reserve_margin(
    make_strategy, mock_exchange, caplog
):
    mock_exchange.capabilities = FuturesCapabilities(
        reserves_margin_on_pending_orders=False
    )
    strategy = make_strategy([_PROFILE])

    with caplog.at_level(logging.INFO):
        asyncio.run(strategy.setup(StrategyRequirements()))

    assert _NOTICE not in caplog.text
