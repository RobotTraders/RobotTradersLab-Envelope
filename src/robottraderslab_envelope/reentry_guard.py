import logging
from dataclasses import dataclass
from datetime import datetime

from robottraderslab import Symbol, TimeFrame
from robottraderslab.strategies import Execution, OHLCVs, OrderSide, PositionSide

logger = logging.getLogger(__name__)

_CLEARANCE_DIRECTION = {PositionSide.LONG: "above", PositionSide.SHORT: "below"}


@dataclass(frozen=True, slots=True)
class ReentryState:
    """Per-symbol re-entry snapshot.

    After a stop-loss fires on a side, that side is added to `blocked_sides`
    and new entries on that side are blocked until price closes back across
    the moving average in the opposite direction. Long and short lock
    independently, so the set can contain either, both, or neither.
    """

    blocked_sides: frozenset[PositionSide]

    def is_entry_allowed(self, side: PositionSide) -> bool:
        return side not in self.blocked_sides


class ReentryGuard:
    """Blocks re-entry after a stop-loss until price closes back across the reference.

    A side is blocked when the most recent stop-loss fill on that side has not
    yet been followed by a close on the opposite side of the moving-average
    reference.
    """

    def __init__(self) -> None:
        self._last_blocked: dict[Symbol, frozenset[PositionSide]] = {}

    def state_for(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        reference_column: str,
        ohlcvs: OHLCVs,
        fills: list[Execution],
        timestamp: datetime,
    ) -> ReentryState:
        """Return the current re-entry state for the symbol.

        Args:
            reference_column: Column carrying the price a stopped-out side
                must clear before it may be entered again.
            ohlcvs: Candles covering the window the fills fall in.
            fills: Stop-loss fills covering at least the loaded window; fills
                for other symbols or before the window are ignored.
            timestamp: The moment the current candle closed. Candles closing
                after it are ignored.
        """
        symbol_fills = [
            fill
            for fill in fills
            if fill.symbol == symbol
            and ohlcvs.covers(symbol, timeframe, fill.timestamp)
        ]
        if not symbol_fills:
            return self._record_state(symbol, frozenset(), {}, timestamp)

        latest_by_side: dict[PositionSide, Execution] = {}
        for fill in symbol_fills:
            side = _closed_position_side(fill)
            current = latest_by_side.get(side)
            if current is None or fill.timestamp > current.timestamp:
                latest_by_side[side] = fill

        blocked = frozenset(
            side
            for side, fill in latest_by_side.items()
            if not _has_cleared(
                ohlcvs, symbol, timeframe, reference_column, fill, timestamp
            )
        )
        return self._record_state(symbol, blocked, latest_by_side, timestamp)

    def _record_state(
        self,
        symbol: Symbol,
        blocked: frozenset[PositionSide],
        latest_by_side: dict[PositionSide, Execution],
        timestamp: datetime,
    ) -> ReentryState:
        previous = self._last_blocked.get(symbol, frozenset())
        for side in sorted(blocked - previous, key=lambda s: s.value):
            logger.info(
                f"{symbol}: {side.value} entries suspended after stop-loss at "
                f"{latest_by_side[side].timestamp}, waiting for a close back "
                f"{_CLEARANCE_DIRECTION[side]} the moving average"
            )
        for side in sorted(previous - blocked, key=lambda s: s.value):
            logger.info(
                f"{symbol}: {side.value} entries resumed at {timestamp}, close "
                f"crossed back {_CLEARANCE_DIRECTION[side]} the moving average"
            )
        self._last_blocked[symbol] = blocked
        return ReentryState(blocked_sides=blocked)


def _closed_position_side(fill: Execution) -> PositionSide:
    """A reducing execution sells to close a long and buys to close a short."""
    return PositionSide.SHORT if fill.side == OrderSide.BUY else PositionSide.LONG


def _has_cleared(
    ohlcvs: OHLCVs,
    symbol: Symbol,
    timeframe: TimeFrame,
    reference_column: str,
    fill: Execution,
    boundary: datetime,
) -> bool:
    """Report whether price closed back across the reference after the fill."""
    since_fill = ohlcvs.candles_between(
        symbol, timeframe, boundary, after=fill.timestamp
    )
    closes = ohlcvs.column(symbol, timeframe, "close")[since_fill]
    references = ohlcvs.column(symbol, timeframe, reference_column)[since_fill]
    if _closed_position_side(fill) == PositionSide.LONG:
        return bool((closes > references).any())
    return bool((closes < references).any())
