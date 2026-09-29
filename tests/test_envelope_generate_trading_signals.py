from collections.abc import Callable

import pandas as pd
import pytest

from robottraderslab import Symbol
from robottraderslab.exceptions import MissingOhlcvDataError
from robottraderslab.strategies import OHLCVs, TradingMode
from robottraderslab_envelope import EnvelopeStrategy

BTC = Symbol.create("BTC/USDT:USDT")
TIMEFRAME = "1d"
SETUP_ID = "BTC/USDT:USDT"


def _build_ohlcvs(
    *,
    close: list[float],
    high: list[float] | None = None,
    low: list[float] | None = None,
) -> OHLCVs:
    length = len(close)
    high = high if high is not None else [c + 1 for c in close]
    low = low if low is not None else [c - 1 for c in close]
    dates = pd.date_range("2024-01-01", periods=length, freq="1D")
    df = pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": [1000.0] * length,
        },
        index=dates,
    )
    return OHLCVs({TIMEFRAME: {BTC: df}})


def _profile(
    *,
    average_type: str = "SMA",
    average_period: int = 3,
    envelopes: list[float] | None = None,
) -> dict:
    return {
        "symbol": "BTC/USDT:USDT",
        "timeframe": TIMEFRAME,
        "total_balance_ratio": 1.0,
        "average_type": average_type,
        "average_period": average_period,
        "envelopes": envelopes if envelopes is not None else [0.05, 0.10],
        "stop_loss_pct": 0.04,
    }


def _collect_values(ohlcvs: OHLCVs, column: str) -> list[float]:
    values: list[float] = []
    for _ in ohlcvs._iter_timeframes():
        values.append(ohlcvs.current(BTC, TIMEFRAME, column))
    return values


@pytest.fixture
def strategy_factory(
    make_strategy: Callable[..., EnvelopeStrategy],
) -> Callable[[dict], EnvelopeStrategy]:
    def _make(profile_dict: dict) -> EnvelopeStrategy:
        return make_strategy([profile_dict])

    return _make


def _ohlcvs_without_the_profile_pair() -> OHLCVs:
    eth = Symbol.create("ETH/USDT:USDT")
    dates = pd.date_range("2024-01-01", periods=5, freq="1D")
    frame = pd.DataFrame(
        {
            "open": [1.0] * 5,
            "high": [2.0] * 5,
            "low": [0.5] * 5,
            "close": [1.0] * 5,
            "volume": [1000.0] * 5,
        },
        index=dates,
    )
    return OHLCVs({TIMEFRAME: {eth: frame}})


class TestGenerateTradingSignals:
    def test_a_missing_pair_propagates_in_backtest(self, strategy_factory):
        strategy = strategy_factory(_profile())
        ohlcvs = _ohlcvs_without_the_profile_pair()

        with pytest.raises(MissingOhlcvDataError):
            strategy.generate_trading_signals(ohlcvs)

    def test_a_missing_pair_skips_the_profile_in_live(self, make_strategy, caplog):
        strategy = make_strategy([_profile()], trading_mode=TradingMode.LIVE)
        ohlcvs = _ohlcvs_without_the_profile_pair()

        strategy.generate_trading_signals(ohlcvs)

        assert "sat out this candle" in caplog.text

    def test_adds_reference_and_band_columns_per_envelope(self, strategy_factory):
        strategy = strategy_factory(
            _profile(envelopes=[0.05, 0.10], average_period=3, average_type="SMA")
        )
        ohlcvs = _build_ohlcvs(close=[100.0, 102.0, 104.0, 106.0])

        strategy.generate_trading_signals(ohlcvs)

        for column in (
            f"{SETUP_ID}_reference",
            f"{SETUP_ID}_band_low_1",
            f"{SETUP_ID}_band_high_1",
            f"{SETUP_ID}_band_low_2",
            f"{SETUP_ID}_band_high_2",
        ):
            assert len(_collect_values(ohlcvs, column)) == 4

    def test_sma_reference_matches_rolling_mean(self, strategy_factory):
        strategy = strategy_factory(_profile(average_type="SMA", average_period=3))
        ohlcvs = _build_ohlcvs(close=[100.0, 102.0, 104.0, 106.0])

        strategy.generate_trading_signals(ohlcvs)

        references = _collect_values(ohlcvs, f"{SETUP_ID}_reference")
        assert references[2] == pytest.approx(102.0)
        assert references[3] == pytest.approx(104.0)

    def test_ema_reference_uses_exponential_average(self, strategy_factory):
        strategy = strategy_factory(_profile(average_type="EMA", average_period=3))
        ohlcvs = _build_ohlcvs(close=[100.0, 100.0, 100.0, 100.0])

        strategy.generate_trading_signals(ohlcvs)

        references = _collect_values(ohlcvs, f"{SETUP_ID}_reference")
        assert references[3] == pytest.approx(100.0)

    def test_wma_reference_uses_weighted_average(self, strategy_factory):
        strategy = strategy_factory(_profile(average_type="WMA", average_period=3))
        ohlcvs = _build_ohlcvs(close=[100.0, 102.0, 104.0])

        strategy.generate_trading_signals(ohlcvs)

        references = _collect_values(ohlcvs, f"{SETUP_ID}_reference")
        expected = (100.0 * 1 + 102.0 * 2 + 104.0 * 3) / 6
        assert references[2] == pytest.approx(expected)

    def test_dcm_reference_uses_high_and_low(self, strategy_factory):
        strategy = strategy_factory(_profile(average_type="DCM", average_period=3))
        ohlcvs = _build_ohlcvs(
            close=[100.0, 100.0, 100.0],
            high=[105.0, 110.0, 108.0],
            low=[95.0, 90.0, 92.0],
        )

        strategy.generate_trading_signals(ohlcvs)

        references = _collect_values(ohlcvs, f"{SETUP_ID}_reference")
        expected = (max(105.0, 110.0, 108.0) + min(95.0, 90.0, 92.0)) / 2
        assert references[2] == pytest.approx(expected)

    def test_bands_fall_the_offset_from_the_reference(self, strategy_factory):
        strategy = strategy_factory(_profile(envelopes=[0.10], average_period=1))
        ohlcvs = _build_ohlcvs(close=[100.0])

        strategy.generate_trading_signals(ohlcvs)

        references = _collect_values(ohlcvs, f"{SETUP_ID}_reference")
        band_lows = _collect_values(ohlcvs, f"{SETUP_ID}_band_low_1")
        band_highs = _collect_values(ohlcvs, f"{SETUP_ID}_band_high_1")
        assert band_lows[0] == pytest.approx(references[0] * 0.90)
        assert band_highs[0] == pytest.approx(references[0] / 0.90)
