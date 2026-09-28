from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from robottraderslab import BotConfig, run_backtest

DATA_DIR = Path(__file__).parent / "data"
CANDLES_FILE = DATA_DIR / "btc-usdt-1h.csv"
TRADINGVIEW_FILE = DATA_DIR / "tradingview-trades.csv"
FIRST_TRADE = 5
LAST_TRADE = 39
CANDLE = pd.Timedelta(hours=1)
PRICE_TOLERANCE = 0.1
MONEY_TOLERANCE = 0.1
RATIO_TOLERANCE = 0.01

TRADINGVIEW_EXIT_REASONS = {
    "Exit": "moving average reached",
    "SL Long": "stop-loss",
    "SL Short": "stop-loss",
}


@dataclass(frozen=True)
class Divergence:
    """A trade value where the engine and TradingView part for a measured
    cause, so the engine's own value is the expected one.
    """

    trade_number: int
    column: str
    engine_value: float
    label: str
    cause: str


TRIGGER_WAITS_FOR_BAND = (
    "The short band (65429.95) already sat below the close (65674.9) when the "
    "rung was placed: TradingView's limit is marketable and fills at the next "
    "open, 65674.9; the engine's trigger waits for the price to fall back to "
    "the band and fills at 65429.95."
)
RUNG_TWO_SIZED_ON_WALLET = (
    "TradingView sizes rung 2 on equity holding rung 1's open PnL at the close "
    "before the fill; the engine sizes it on the wallet balance, realised PnL "
    "only."
)

DIVERGENCES = (
    Divergence(
        37, "entry_price", 65429.95, "trigger-waits-for-band", TRIGGER_WAITS_FOR_BAND
    ),
    Divergence(37, "net_pnl", 2.31, "trigger-waits-for-band", TRIGGER_WAITS_FOR_BAND),
    Divergence(
        26,
        "net_pnl",
        19.13,
        "rung-2-sized-on-wallet",
        RUNG_TWO_SIZED_ON_WALLET
        + " TradingView 1036.94 = 1034.05 realised + 2.89 open; engine 1033.74.",
    ),
    Divergence(
        34,
        "net_pnl",
        9.26,
        "rung-2-sized-on-wallet",
        RUNG_TWO_SIZED_ON_WALLET
        + " TradingView 1054.10 = 1072.95 realised - 18.85 open; engine 1072.56.",
    ),
)

MOVED_BY_DIVERGENCES = "moved-by-divergences"


def _trade_params(column: str) -> list[object]:
    labels = {
        divergence.trade_number: divergence.label
        for divergence in DIVERGENCES
        if divergence.column == column
    }
    return [
        pytest.param(number, id=f"{number}-{labels[number]}")
        if number in labels
        else pytest.param(number, id=str(number))
        for number in range(FIRST_TRADE, LAST_TRADE + 1)
    ]


@pytest.fixture(scope="module")
def engine_trades() -> pd.DataFrame:
    """The run's trades in TradingView's order: by entry, then by rung.

    Module scope, since the backtest is one run every check reads.
    """
    outputs = run_backtest(BotConfig.from_text(_config_text()))
    trades = outputs.create_analyser(save=False).trades
    return _normalise_engine_trades(trades)


@pytest.fixture(scope="module")
def expected_trades() -> pd.DataFrame:
    """The export's trades in the window, one row each, stamped the way the
    engine stamps a fill, at the close of the candle it happens in, and
    carrying the engine's value wherever a divergence names one.
    """
    export = pd.read_csv(TRADINGVIEW_FILE, encoding="utf-8-sig")
    window = export[export["Trade number"].between(FIRST_TRADE, LAST_TRADE)]
    expected = _normalise_tradingview_trades(window)
    for divergence in DIVERGENCES:
        expected.loc[divergence.trade_number, divergence.column] = (
            divergence.engine_value
        )
    return expected


@pytest.fixture
def engine_figures(engine_trades: pd.DataFrame) -> dict[str, float]:
    return _window_figures(engine_trades)


@pytest.fixture
def expected_figures(expected_trades: pd.DataFrame) -> dict[str, float]:
    return _window_figures(expected_trades)


class TestEveryTrade:
    def test_trade_count(self, engine_trades, expected_trades):
        assert len(engine_trades) == len(expected_trades)

    @pytest.mark.parametrize("trade_number", _trade_params("side"))
    def test_side(self, trade_number, engine_trades, expected_trades):
        assert (
            engine_trades.loc[trade_number, "side"]
            == expected_trades.loc[trade_number, "side"]
        )

    @pytest.mark.parametrize("trade_number", _trade_params("rung"))
    def test_rung(self, trade_number, engine_trades, expected_trades):
        assert (
            engine_trades.loc[trade_number, "rung"]
            == expected_trades.loc[trade_number, "rung"]
        )

    @pytest.mark.parametrize("trade_number", _trade_params("entry_time"))
    def test_entry_candle(self, trade_number, engine_trades, expected_trades):
        assert (
            engine_trades.loc[trade_number, "entry_time"]
            == expected_trades.loc[trade_number, "entry_time"]
        )

    @pytest.mark.parametrize("trade_number", _trade_params("exit_time"))
    def test_exit_candle(self, trade_number, engine_trades, expected_trades):
        assert (
            engine_trades.loc[trade_number, "exit_time"]
            == expected_trades.loc[trade_number, "exit_time"]
        )

    @pytest.mark.parametrize("trade_number", _trade_params("entry_price"))
    def test_entry_price(self, trade_number, engine_trades, expected_trades):
        assert engine_trades.loc[trade_number, "entry_price"] == pytest.approx(
            expected_trades.loc[trade_number, "entry_price"], abs=PRICE_TOLERANCE
        )

    @pytest.mark.parametrize("trade_number", _trade_params("exit_price"))
    def test_exit_price(self, trade_number, engine_trades, expected_trades):
        assert engine_trades.loc[trade_number, "exit_price"] == pytest.approx(
            expected_trades.loc[trade_number, "exit_price"], abs=PRICE_TOLERANCE
        )

    @pytest.mark.parametrize("trade_number", _trade_params("exit_reason"))
    def test_exit_reason(self, trade_number, engine_trades, expected_trades):
        assert (
            engine_trades.loc[trade_number, "exit_reason"]
            == expected_trades.loc[trade_number, "exit_reason"]
        )

    @pytest.mark.parametrize("trade_number", _trade_params("net_pnl"))
    def test_net_pnl(self, trade_number, engine_trades, expected_trades):
        assert engine_trades.loc[trade_number, "net_pnl"] == pytest.approx(
            expected_trades.loc[trade_number, "net_pnl"], abs=MONEY_TOLERANCE
        )

    @pytest.mark.parametrize("trade_number", _trade_params("commission"))
    def test_commission(self, trade_number, engine_trades, expected_trades):
        assert engine_trades.loc[trade_number, "commission"] == pytest.approx(
            expected_trades.loc[trade_number, "commission"], abs=MONEY_TOLERANCE
        )

    @pytest.mark.parametrize("trade_number", _trade_params("duration"))
    def test_duration(self, trade_number, engine_trades, expected_trades):
        assert (
            engine_trades.loc[trade_number, "duration"]
            == expected_trades.loc[trade_number, "duration"]
        )


class TestWindowFigures:
    @pytest.mark.parametrize(
        ("figure", "tolerance"),
        [
            ("trades", 0),
            ("longs", 0),
            ("shorts", 0),
            ("winners", 0),
            ("losers", 0),
            ("win_rate", RATIO_TOLERANCE),
            pytest.param(
                "gross_profit",
                MONEY_TOLERANCE,
                id=f"gross_profit-{MOVED_BY_DIVERGENCES}",
            ),
            ("gross_loss", MONEY_TOLERANCE),
            pytest.param(
                "profit_factor",
                RATIO_TOLERANCE,
                id=f"profit_factor-{MOVED_BY_DIVERGENCES}",
            ),
            ("largest_win", MONEY_TOLERANCE),
            ("largest_loss", MONEY_TOLERANCE),
            pytest.param(
                "average_win", MONEY_TOLERANCE, id=f"average_win-{MOVED_BY_DIVERGENCES}"
            ),
            ("average_loss", MONEY_TOLERANCE),
            ("total_commission", MONEY_TOLERANCE),
            pytest.param(
                "net_pnl_rung_1",
                MONEY_TOLERANCE,
                id=f"net_pnl_rung_1-{MOVED_BY_DIVERGENCES}",
            ),
            pytest.param(
                "net_pnl_rung_2",
                MONEY_TOLERANCE,
                id=f"net_pnl_rung_2-{MOVED_BY_DIVERGENCES}",
            ),
        ],
    )
    def test_figure(self, figure, tolerance, engine_figures, expected_figures):
        assert engine_figures[figure] == pytest.approx(
            expected_figures[figure], abs=tolerance
        )


def _config_text() -> str:
    return f"""\
[strategy]
strategy_class = "envelope"

[[strategy.profiles]]
symbol = "BTC/USDT:USDT"
timeframe = "1h"
average_type = "DCM"
average_period = 5
envelopes = [0.04, 0.06]
stop_loss_pct = 0.25
total_balance_ratio = 1.0
leverage = 1.0
margin_mode = "isolated"

[backtest]
initial_balance = {{ USDT = 1015.74 }}
maker_fee_rate = 0.0002
taker_fee_rate = 0.0002
placement_reserve = {{ margin_markup = 0.0, fee_markup = 0.0, notional_reserve = 0.0 }}
start_date = "2025-01-05"
end_date = "2026-02-10"

[backtest.ohlcv_provider]
ohlcv_provider = "csv"
file = "{CANDLES_FILE.as_posix()}"
symbol = "BTC/USDT:USDT"
timeframe = "1h"
"""


def _normalise_engine_trades(trades: pd.DataFrame) -> pd.DataFrame:
    rungs = trades["entry_reason"].str.extract(r"entry rung (\d+)")[0].astype(int)
    normalised = pd.DataFrame(
        {
            "side": trades["side"],
            "rung": rungs,
            "entry_time": trades["entry_time"],
            "exit_time": trades["exit_time"],
            "entry_price": trades["entry_price"],
            "exit_price": trades["exit_price"],
            "exit_reason": trades["exit_reason"],
            "net_pnl": trades["net_pnl"],
            "commission": trades["entry_fee"] + trades["exit_fee"],
            "duration": (trades["exit_time"] - trades["entry_time"]) // CANDLE,
        }
    ).sort_values(["entry_time", "rung"])
    normalised.index = pd.RangeIndex(FIRST_TRADE, FIRST_TRADE + len(normalised))
    return normalised


def _normalise_tradingview_trades(export: pd.DataFrame) -> pd.DataFrame:
    entries = export[export["Type"].str.startswith("Entry")].set_index("Trade number")
    exits = export[export["Type"].str.startswith("Exit")].set_index("Trade number")
    return pd.DataFrame(
        {
            "side": entries["Type"].str.removeprefix("Entry "),
            "rung": entries["Signal"].str[-1].astype(int),
            "entry_time": _engine_stamp(entries["Date and time"]),
            "exit_time": _engine_stamp(exits["Date and time"]),
            "entry_price": entries["Price USDT"],
            "exit_price": exits["Price USDT"],
            "exit_reason": exits["Signal"].map(TRADINGVIEW_EXIT_REASONS),
            "net_pnl": entries["Net PnL USDT"],
            "commission": entries["Commission USDT"],
            "duration": entries["Duration (bars)"],
        }
    ).sort_index()


def _engine_stamp(tradingview_times: pd.Series) -> pd.Series:
    return pd.to_datetime(tradingview_times, utc=True) + CANDLE


def _window_figures(trades: pd.DataFrame) -> dict[str, float]:
    net_pnl = trades["net_pnl"]
    wins = net_pnl[net_pnl > 0]
    losses = net_pnl[net_pnl <= 0]
    return {
        "trades": len(trades),
        "longs": int((trades["side"] == "long").sum()),
        "shorts": int((trades["side"] == "short").sum()),
        "winners": len(wins),
        "losers": len(losses),
        "win_rate": len(wins) / len(trades),
        "gross_profit": wins.sum(),
        "gross_loss": losses.sum(),
        "profit_factor": wins.sum() / -losses.sum(),
        "largest_win": wins.max(),
        "largest_loss": losses.min(),
        "average_win": wins.mean(),
        "average_loss": losses.mean(),
        "total_commission": trades["commission"].sum(),
        "net_pnl_rung_1": net_pnl[trades["rung"] == 1].sum(),
        "net_pnl_rung_2": net_pnl[trades["rung"] == 2].sum(),
    }
