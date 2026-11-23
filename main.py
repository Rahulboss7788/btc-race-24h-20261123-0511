"""Breakout-surfer: a trend / volatility-breakout bot for cbBTC/USDC.

Strategy, in one paragraph: the bot sits in cash until the market proves a
real uptrend, then rides it with a wide trailing stop. A trade starts only on
a *fresh* breakout -- the first close to jump clear above the 120-bar
Donchian high, out of a tight range, while the fast EMA runs above the slow
EMA and short-horizon volatility is both expanding and above its long-run
level. Once long, the bot exits on a wide ATR trailing stop (1.2% of price
from the peak), on the fast EMA crossing back under the slow EMA, on a 1.5%
hard stop, on a 600-bar time stop, or -- if the breakout never follows
through (less than 0.4% above the entry after 30 bars) -- on a follow-through
failure. If no strict setup ever appears, a relaxed setup is allowed in the
final 6 hours so the bot still takes one disciplined shot instead of sitting
out the whole window. One position at a time, full size, long only.

The bot speaks agent-fun/bars-replay/v1 through the vendored SDK shim
(vendor/makefun_bars.py). It is fully deterministic: no clock, no environment,
no network, no randomness -- every decision is a pure function of the bars
seen so far, so two runs on the same data produce identical action logs.
Only the Python standard library is used.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

SYMBOL = "CBBTC"

# --- trend filter -----------------------------------------------------------
EMA_FAST = 12          # fast EMA span, minutes
EMA_SLOW = 48          # slow EMA span, minutes
TREND_MIN_BPS = 8      # fast-minus-slow EMA gap needed to call it a trend
TREND_BARS = 5         # consecutive bars the EMA gap must hold before entry

# --- breakout filter --------------------------------------------------------
DONCHIAN_BARS = 120    # range high/low lookback, minutes
BREAK_MARGIN_BPS = 15  # close must clear the range high by this much
RANGE_MAX_BPS = 60     # range (high-low) must be tighter than this to qualify

# --- volatility regime ------------------------------------------------------
ATR_BARS = 20          # short ATR window, minutes
ATR_LONG_BARS = 120    # long ATR window, minutes
ATR_MIN_BPS = 3        # ignore dead-quiet bars
ATR_MAX_BPS = 80       # ignore blow-out volatility
VOL_RATIO = 1.15       # short ATR must be at least this x the long ATR

# --- exits ------------------------------------------------------------------
TRAIL_ATR_MULT = 6.0   # trailing stop: peak minus this many ATRs ...
TRAIL_MIN_BPS = 120    # ... but never less than this far under the peak
STOP_BPS = 150         # hard stop under the entry price
MAX_HOLD_BARS = 600    # time stop
FOLLOW_BARS = 30       # follow-through check after this many bars held
FOLLOW_MIN_BPS = 40    # ... needs at least this much peak gain to stay in

# --- after an exit ----------------------------------------------------------
COOLDOWN_BARS = 60     # wait this long before the next entry

# --- late-window relaxed setup (only while no trade has filled yet) ---------
LATE_SEQ = 1080        # from this batch on, if still untraded
LATE_DONCHIAN = 60
LATE_MARGIN_BPS = 0    # any new high counts
LATE_RANGE_BPS = 150
LATE_VOL_RATIO = 1.0
LATE_TREND_BPS = 5

MAX_SLIPPAGE_BPS = 300  # orders may fill up to 3% away from the last close


class BreakoutSurfer:
    """Keeps the indicator state and decides one reply's orders per batch."""

    def __init__(self):
        self.closes = []
        self.highs = []
        self.lows = []
        self.true_ranges = []
        self.ema_fast = None
        self.ema_slow = None
        self.trend_run = 0
        self.in_position = False
        self.entry_price = 0
        self.peak_price = 0
        self.hold_bars = 0
        self.cooldown = 0
        self.filled_trades = 0

    # -- indicator updates ---------------------------------------------------
    def _ema_step(self, previous, value, span):
        if previous is None:
            return float(value)
        return previous + (2.0 / (span + 1)) * (value - previous)

    def _atr(self, window):
        recent = self.true_ranges[-window:]
        return sum(recent) / len(recent)

    def _donchian(self, window):
        # High/low over `window` bars ending at the previous bar (the current
        # bar is still forming its own range and must not set the bar to beat).
        highs = self.highs[-window - 1:-1]
        lows = self.lows[-window - 1:-1]
        return max(highs), min(lows)

    # -- batch handling ------------------------------------------------------
    def on_batch(self, state, batch):
        bar = None
        for candidate in batch.get("bars", []):
            if candidate.get("symbol") == SYMBOL:
                bar = candidate
                break
        if bar is None:
            return []  # this minute carried no CBBTC bar: nothing new to do

        close = int(bar["c"])
        high = int(bar["h"])
        low = int(bar["l"])

        if self.closes:
            prev_close = self.closes[-1]
            true_range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        else:
            true_range = high - low
        self.closes.append(close)
        self.highs.append(high)
        self.lows.append(low)
        self.true_ranges.append(true_range)
        self.ema_fast = self._ema_step(self.ema_fast, close, EMA_FAST)
        self.ema_slow = self._ema_step(self.ema_slow, close, EMA_SLOW)

        if self.cooldown > 0:
            self.cooldown -= 1

        # Sync the position with what the harness reports, including fills.
        holding = state.positions.get(SYMBOL)
        held_qty = holding.qty if holding is not None else 0
        for fill in batch.get("fills", []):
            if fill.get("symbol") != SYMBOL or fill.get("status") != "filled":
                continue
            self.filled_trades += 1
            if fill.get("side") == "buy":
                self.in_position = True
                self.entry_price = int(fill["price"])
                self.peak_price = self.entry_price
                self.hold_bars = 0
        if held_qty > 0 and not self.in_position:
            # Position present but no fill seen (defensive): derive the entry
            # from the position's cost, fees included.
            self.in_position = True
            self.entry_price = holding.cost * 10 ** 7 // held_qty
            self.peak_price = self.entry_price
            self.hold_bars = 0
        if held_qty == 0 and self.in_position:
            self.in_position = False
            self.cooldown = COOLDOWN_BARS
        if self.in_position:
            self.hold_bars += 1
            if close > self.peak_price:
                self.peak_price = close

        if state.warming_up:
            return []  # warm-up bars build history only; orders are rejected

        need = max(EMA_SLOW, DONCHIAN_BARS, ATR_LONG_BARS) + 20
        if len(self.closes) < need:
            return []  # not enough history to judge a breakout yet

        if self.in_position:
            return self._exit_orders(state, close, held_qty)
        return self._entry_orders(state, close)

    # -- exits ---------------------------------------------------------------
    def _exit_orders(self, state, close, held_qty):
        atr = self._atr(ATR_BARS)
        trail = self.peak_price - max(
            TRAIL_ATR_MULT * atr, TRAIL_MIN_BPS * self.peak_price / 10000
        )
        peak_gain_bps = (self.peak_price - self.entry_price) * 10000 // self.entry_price
        should_exit = (
            close <= trail
            or self.ema_fast < self.ema_slow
            or close <= self.entry_price * (10000 - STOP_BPS) // 10000
            or (self.hold_bars >= FOLLOW_BARS and peak_gain_bps < FOLLOW_MIN_BPS)
            or self.hold_bars >= MAX_HOLD_BARS
        )
        if not should_exit:
            return []
        return [{
            "side": "sell",
            "symbol": SYMBOL,
            "qty": held_qty,
            "max_slippage_bps": MAX_SLIPPAGE_BPS,
        }]

    # -- entries -------------------------------------------------------------
    def _entry_orders(self, state, close):
        # Track the trend run before any early return, so it keeps counting
        # through a cooldown (matching the flat-state update order): a trend
        # that started while cooling down is already confirmed when the
        # cooldown ends.
        if self.ema_fast > self.ema_slow:
            self.trend_run += 1
        else:
            self.trend_run = 0
        if self.cooldown > 0 or state.cash <= 0:
            return []
        relaxed = self.filled_trades == 0 and state.seq >= LATE_SEQ
        donchian = LATE_DONCHIAN if relaxed else DONCHIAN_BARS
        margin = LATE_MARGIN_BPS if relaxed else BREAK_MARGIN_BPS
        range_max = LATE_RANGE_BPS if relaxed else RANGE_MAX_BPS
        vol_ratio = LATE_VOL_RATIO if relaxed else VOL_RATIO
        trend_min = LATE_TREND_BPS if relaxed else TREND_MIN_BPS

        atr = self._atr(ATR_BARS)
        atr_long = self._atr(ATR_LONG_BARS)
        atr_bps = atr / close * 10000
        atr_ago = self._atr_window_ago(ATR_BARS, 20)
        expanding = atr >= atr_ago
        vol_ok = atr >= vol_ratio * atr_long

        trend_strength_bps = (self.ema_fast - self.ema_slow) / close * 10000

        range_high, range_low = self._donchian(donchian)
        range_bps = (range_high - range_low) * 10000 // close
        # First close above the breakout threshold: the previous close must
        # not already have cleared it, so a stalled breakout is not chased.
        fresh = self.closes[-2] <= range_high * (10000 + margin) // 10000
        breakout = close >= range_high * (10000 + margin) // 10000

        if (
            trend_strength_bps >= trend_min
            and self.trend_run >= TREND_BARS
            and breakout
            and fresh
            and range_bps <= range_max
            and expanding
            and vol_ok
            and ATR_MIN_BPS <= atr_bps <= ATR_MAX_BPS
        ):
            return [{
                "side": "buy",
                "symbol": SYMBOL,
                "notional_quote_units": state.cash,
                "max_slippage_bps": MAX_SLIPPAGE_BPS,
            }]
        return []

    def _atr_window_ago(self, window, bars_ago):
        # Mean true range over `window` bars ending `bars_ago` bars back.
        end = len(self.true_ranges) - bars_ago
        if end <= 0:
            return self._atr(window)
        start = max(0, end - window)
        recent = self.true_ranges[start:end]
        return sum(recent) / len(recent)


def main():
    bot = BreakoutSurfer()
    makefun_bars.run(bot.on_batch)


if __name__ == "__main__":
    main()
