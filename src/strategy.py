"""
Strategy framework with shared trade models and concrete implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import List, Literal, Optional, Tuple

import pandas as pd

from .indicators import (
    average_directional_index,
    average_true_range,
    bollinger_bands,
    exponential_moving_average,
    macd,
    relative_strength_index,
    simple_moving_average,
    stochastic_oscillator,
    volume_moving_average,
    williams_r,
)

TradeDirection = Literal["long", "short"]
TradeExit = Literal["take_profit", "stop_loss"]
StopMode = Literal["atr", "swing"]
TradePreference = Literal["long", "short", "both"]


@dataclass
class StrategySettings:
    """
    General-purpose configuration shared by the concrete strategies.
    """

    session_open: time = time(0, 0)
    range_duration: timedelta = timedelta(hours=4)
    atr_period: int = 14
    reward_multiple: float = 2.0
    session_timezone: timezone = timezone.utc
    sma_period: Optional[int] = None
    stop_mode: StopMode = "atr"
    swing_stop_period: int = 20
    trade_directions: TradePreference = "both"
    capital_base: float = 1.0
    position_fraction: float = 1.0
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    ema_fast_period: int = 12
    ema_slow_period: int = 26
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    volume_ma_period: int = 20
    volume_multiplier: float = 1.5
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_oversold: float = 20.0
    stoch_overbought: float = 80.0
    adx_period: int = 14
    adx_threshold: float = 25.0
    willr_period: int = 14
    willr_oversold: float = -80.0
    willr_overbought: float = -20.0


@dataclass
class Trade:
    """Represents a single trade outcome."""

    symbol: str
    direction: TradeDirection
    entry_time: datetime
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    risk_value: float
    risk_type: StopMode
    cost_multiplier: float = 1.0
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[TradeExit] = None
    notes: str = ""
    fees_paid: float = 0.0

    def pnl(self) -> Optional[float]:
        """Return absolute profit or loss if the trade has closed."""
        if self.exit_price is None or self.exit_reason is None:
            return None

        if self.direction == "long":
            raw = (self.exit_price - self.entry_price) * self.quantity
        else:
            raw = (self.entry_price - self.exit_price) * self.quantity

        return raw * self.cost_multiplier

    def holding_period(self) -> Optional[timedelta]:
        if self.exit_time is None:
            return None
        return self.exit_time - self.entry_time


class BaseStrategy:
    """Base class with shared indicator preparation and trade utilities."""

    def __init__(self, symbol: str, settings: Optional[StrategySettings] = None):
        self.symbol = symbol
        self.settings = settings or StrategySettings()

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Compute shared indicators used across strategies."""
        frame = frame.sort_index().copy()

        if self.settings.stop_mode == "atr":
            average_true_range(frame, period=self.settings.atr_period, column_name="atr")
        elif self.settings.stop_mode == "swing":
            period = max(1, self.settings.swing_stop_period)
            frame["swing_low"] = frame["low"].rolling(window=period, min_periods=period).min()
            frame["swing_high"] = frame["high"].rolling(window=period, min_periods=period).max()
        else:
            raise ValueError(f"Unsupported stop mode: {self.settings.stop_mode}")

        if self.settings.sma_period:
            simple_moving_average(frame, period=self.settings.sma_period, column_name="sma")

        return frame

    def _group_by_session(self, frame: pd.DataFrame):
        local_frame = frame.tz_convert(self.settings.session_timezone)
        return local_frame.groupby(local_frame.index.date)

    def _day_bounds(self, day: datetime.date) -> Tuple[datetime, datetime]:
        start = datetime.combine(day, self.settings.session_open, tzinfo=self.settings.session_timezone)
        end = start + timedelta(days=1)
        return start, end

    def _compute_risk_distance(
        self,
        direction: TradeDirection,
        price: float,
        atr_value: Optional[float],
        swing_low: Optional[float],
        swing_high: Optional[float],
    ) -> Optional[float]:
        if self.settings.stop_mode == "atr":
            if atr_value is None or pd.isna(atr_value):
                return None
            if direction == "long" and self.settings.trade_directions not in ("long", "both"):
                return None
            if direction == "short" and self.settings.trade_directions not in ("short", "both"):
                return None
            return float(atr_value)

        if self.settings.stop_mode == "swing":
            if swing_low is None or swing_high is None or pd.isna(swing_low) or pd.isna(swing_high):
                return None
            if direction == "long":
                if self.settings.trade_directions not in ("long", "both"):
                    return None
                return float(price - swing_low)
            if self.settings.trade_directions not in ("short", "both"):
                return None
            return float(swing_high - price)

        raise ValueError(f"Unsupported stop mode: {self.settings.stop_mode}")

    def _position_quantity(self, price: float) -> Optional[float]:
        notional = self.settings.capital_base * self.settings.position_fraction
        if notional <= 0 or price <= 0:
            return None
        return notional / price

    def _manage_active_trade(
        self,
        active_trade: Optional[Trade],
        timestamp: datetime,
        row: pd.Series,
    ) -> Tuple[Optional[Trade], Optional[Trade]]:
        if active_trade is None:
            return None, None

        if active_trade.direction == "long":
            stop_hit = row["low"] <= active_trade.stop_loss
            target_hit = row["high"] >= active_trade.take_profit
        else:
            stop_hit = row["high"] >= active_trade.stop_loss
            target_hit = row["low"] <= active_trade.take_profit

        if not (stop_hit or target_hit):
            return active_trade, None

        active_trade.exit_time = timestamp
        if stop_hit and target_hit:
            # Assume worst-case: stop loss fills first.
            active_trade.exit_price = active_trade.stop_loss
            active_trade.exit_reason = "stop_loss"
        elif stop_hit:
            active_trade.exit_price = active_trade.stop_loss
            active_trade.exit_reason = "stop_loss"
        else:
            active_trade.exit_price = active_trade.take_profit
            active_trade.exit_reason = "take_profit"

        return None, active_trade


class IntradayRangeStrategy(BaseStrategy):
    """Fade the first 4 hour range once price re-enters from outside the extremes."""

    def evaluate(self, frame: pd.DataFrame) -> List[Trade]:
        prepared = self.prepare(frame)
        trades: list[Trade] = []

        for day, day_frame in self._group_by_session(prepared):
            start, end = self._day_bounds(day)
            day_frame = day_frame[(day_frame.index >= start) & (day_frame.index < end)]
            if day_frame.empty:
                continue

            trades.extend(self._evaluate_day(day_frame))

        return trades

    def _evaluate_day(self, day_frame: pd.DataFrame) -> List[Trade]:
        range_end = day_frame.index[0] + self.settings.range_duration
        range_frame = day_frame[day_frame.index < range_end]

        min_period = (
            self.settings.atr_period if self.settings.stop_mode == "atr" else self.settings.swing_stop_period
        )
        if len(range_frame) < max(1, min_period):
            return []

        initial_high = range_frame["high"].max()
        initial_low = range_frame["low"].min()

        above_range = False
        below_range = False
        active_trade: Optional[Trade] = None
        trades: list[Trade] = []

        for timestamp, row in day_frame.iterrows():
            price = row["close"]
            sma_value = row.get("sma")

            active_trade, closed_trade = self._manage_active_trade(active_trade, timestamp, row)
            if closed_trade:
                trades.append(closed_trade)
                above_range = False
                below_range = False
                continue

            if timestamp < range_end:
                continue

            if row["high"] > initial_high:
                above_range = True
            if row["low"] < initial_low:
                below_range = True

            if not active_trade and above_range and price < initial_high:
                if self.settings.sma_period and (pd.isna(sma_value) or price > sma_value):
                    continue

                risk_distance = self._compute_risk_distance(
                    "short", price, row.get("atr"), row.get("swing_low"), row.get("swing_high")
                )
                if risk_distance is None or risk_distance <= 0:
                    continue

                quantity = self._position_quantity(price)
                if quantity is None or quantity <= 0:
                    continue

                entry_price = price
                stop_loss = entry_price + risk_distance
                take_profit = entry_price - self.settings.reward_multiple * risk_distance
                active_trade = Trade(
                    symbol=self.symbol,
                    direction="short",
                    entry_time=timestamp,
                    entry_price=entry_price,
                    quantity=quantity,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_value=risk_distance,
                    risk_type=self.settings.stop_mode,
                    notes="Fade move back inside 4h range high",
                )
                continue

            if not active_trade and below_range and price > initial_low:
                if self.settings.sma_period and (pd.isna(sma_value) or price < sma_value):
                    continue

                risk_distance = self._compute_risk_distance(
                    "long", price, row.get("atr"), row.get("swing_low"), row.get("swing_high")
                )
                if risk_distance is None or risk_distance <= 0:
                    continue

                quantity = self._position_quantity(price)
                if quantity is None or quantity <= 0:
                    continue

                entry_price = price
                stop_loss = entry_price - risk_distance
                take_profit = entry_price + self.settings.reward_multiple * risk_distance
                active_trade = Trade(
                    symbol=self.symbol,
                    direction="long",
                    entry_time=timestamp,
                    entry_price=entry_price,
                    quantity=quantity,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_value=risk_distance,
                    risk_type=self.settings.stop_mode,
                    notes="Fade move back inside 4h range low",
                )
                continue

        if active_trade:
            active_trade.exit_time = day_frame.index[-1]
            active_trade.exit_price = day_frame.iloc[-1]["close"]
            active_trade.exit_reason = "stop_loss"
            trades.append(active_trade)

        return trades


class MomentumBreakoutStrategy(BaseStrategy):
    """Trade momentum when price breaks the previous day's high/low."""

    def evaluate(self, frame: pd.DataFrame) -> List[Trade]:
        prepared = self.prepare(frame)
        trades: list[Trade] = []

        prev_day_high: Optional[float] = None
        prev_day_low: Optional[float] = None

        for day, day_frame in self._group_by_session(prepared):
            start, end = self._day_bounds(day)
            day_frame = day_frame[(day_frame.index >= start) & (day_frame.index < end)]
            if day_frame.empty:
                continue

            day_high = day_frame["high"].max()
            day_low = day_frame["low"].min()

            if prev_day_high is None or prev_day_low is None:
                prev_day_high = day_high
                prev_day_low = day_low
                continue

            trades.extend(self._evaluate_day(day_frame, prev_day_high, prev_day_low))

            prev_day_high = day_high
            prev_day_low = day_low

        return trades

    def _evaluate_day(
        self,
        day_frame: pd.DataFrame,
        prev_day_high: float,
        prev_day_low: float,
    ) -> List[Trade]:
        active_trade: Optional[Trade] = None
        trades: list[Trade] = []
        long_triggered = False
        short_triggered = False

        for timestamp, row in day_frame.iterrows():
            price = row["close"]
            sma_value = row.get("sma")

            active_trade, closed_trade = self._manage_active_trade(active_trade, timestamp, row)
            if closed_trade:
                trades.append(closed_trade)
                continue

            if active_trade:
                continue

            if not long_triggered and row["high"] >= prev_day_high:
                if self.settings.sma_period and (pd.isna(sma_value) or price < sma_value):
                    continue

                risk_distance = self._compute_risk_distance(
                    "long", price, row.get("atr"), row.get("swing_low"), row.get("swing_high")
                )
                if risk_distance is None or risk_distance <= 0:
                    continue

                quantity = self._position_quantity(price)
                if quantity is None or quantity <= 0:
                    continue

                entry_price = price
                stop_loss = entry_price - risk_distance
                take_profit = entry_price + self.settings.reward_multiple * risk_distance
                active_trade = Trade(
                    symbol=self.symbol,
                    direction="long",
                    entry_time=timestamp,
                    entry_price=entry_price,
                    quantity=quantity,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_value=risk_distance,
                    risk_type=self.settings.stop_mode,
                    notes="Breakout above previous day high",
                )
                long_triggered = True
                continue

            if not short_triggered and row["low"] <= prev_day_low:
                if self.settings.sma_period and (pd.isna(sma_value) or price > sma_value):
                    continue

                risk_distance = self._compute_risk_distance(
                    "short", price, row.get("atr"), row.get("swing_low"), row.get("swing_high")
                )
                if risk_distance is None or risk_distance <= 0:
                    continue

                quantity = self._position_quantity(price)
                if quantity is None or quantity <= 0:
                    continue

                entry_price = price
                stop_loss = entry_price + risk_distance
                take_profit = entry_price - self.settings.reward_multiple * risk_distance
                active_trade = Trade(
                    symbol=self.symbol,
                    direction="short",
                    entry_time=timestamp,
                    entry_price=entry_price,
                    quantity=quantity,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_value=risk_distance,
                    risk_type=self.settings.stop_mode,
                    notes="Breakout below previous day low",
                )
                short_triggered = True
                continue

        if active_trade:
            active_trade.exit_time = day_frame.index[-1]
            active_trade.exit_price = day_frame.iloc[-1]["close"]
            active_trade.exit_reason = "stop_loss"
            trades.append(active_trade)

        return trades


class BollingerBandsMeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy using Bollinger Bands.
    
    Enters long when price touches or breaks below the lower band (oversold),
    and enters short when price touches or breaks above the upper band (overbought).
    Exits when price returns toward the middle band or hits stop/target.
    """

    def evaluate(self, frame: pd.DataFrame) -> List[Trade]:
        prepared = self.prepare(frame)
        
        # Add Bollinger Bands
        bollinger_bands(
            prepared,
            period=self.settings.bb_period,
            num_std=self.settings.bb_std,
            sma_column="bb_sma",
            upper_column="bb_upper",
            lower_column="bb_lower",
            middle_column="bb_middle",
        )
        
        trades: list[Trade] = []
        active_trade: Optional[Trade] = None

        for timestamp, row in prepared.iterrows():
            price = row["close"]
            bb_upper = row.get("bb_upper")
            bb_lower = row.get("bb_lower")
            bb_middle = row.get("bb_middle")
            sma_value = row.get("sma")

            # Skip if Bollinger Bands aren't ready
            if pd.isna(bb_upper) or pd.isna(bb_lower) or pd.isna(bb_middle):
                continue

            # Manage active trade
            active_trade, closed_trade = self._manage_active_trade(active_trade, timestamp, row)
            if closed_trade:
                trades.append(closed_trade)
                continue

            # Only enter new trades if we don't have an active one
            if active_trade:
                continue

            # Long entry: price touches or breaks below lower band (oversold bounce)
            if self.settings.trade_directions in ("long", "both"):
                if row["low"] <= bb_lower:
                    # Optional SMA filter: only long if price is above SMA (trend filter)
                    if self.settings.sma_period and (pd.isna(sma_value) or price < sma_value):
                        continue

                    risk_distance = self._compute_risk_distance(
                        "long", price, row.get("atr"), row.get("swing_low"), row.get("swing_high")
                    )
                    if risk_distance is None or risk_distance <= 0:
                        continue

                    quantity = self._position_quantity(price)
                    if quantity is None or quantity <= 0:
                        continue

                    entry_price = price
                    stop_loss = entry_price - risk_distance
                    take_profit = entry_price + self.settings.reward_multiple * risk_distance
                    
                    # Alternative: target middle band if it's closer than ATR-based target
                    middle_target = bb_middle
                    if middle_target > take_profit:
                        take_profit = middle_target

                    active_trade = Trade(
                        symbol=self.symbol,
                        direction="long",
                        entry_time=timestamp,
                        entry_price=entry_price,
                        quantity=quantity,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_value=risk_distance,
                        risk_type=self.settings.stop_mode,
                        notes=f"BB mean reversion: oversold bounce from lower band ({bb_lower:.2f})",
                    )
                    continue

            # Short entry: price touches or breaks above upper band (overbought fade)
            if self.settings.trade_directions in ("short", "both"):
                if row["high"] >= bb_upper:
                    # Optional SMA filter: only short if price is below SMA (trend filter)
                    if self.settings.sma_period and (pd.isna(sma_value) or price > sma_value):
                        continue

                    risk_distance = self._compute_risk_distance(
                        "short", price, row.get("atr"), row.get("swing_low"), row.get("swing_high")
                    )
                    if risk_distance is None or risk_distance <= 0:
                        continue

                    quantity = self._position_quantity(price)
                    if quantity is None or quantity <= 0:
                        continue

                    entry_price = price
                    stop_loss = entry_price + risk_distance
                    take_profit = entry_price - self.settings.reward_multiple * risk_distance
                    
                    # Alternative: target middle band if it's closer than ATR-based target
                    middle_target = bb_middle
                    if middle_target < take_profit:
                        take_profit = middle_target

                    active_trade = Trade(
                        symbol=self.symbol,
                        direction="short",
                        entry_time=timestamp,
                        entry_price=entry_price,
                        quantity=quantity,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_value=risk_distance,
                        risk_type=self.settings.stop_mode,
                        notes=f"BB mean reversion: overbought fade from upper band ({bb_upper:.2f})",
                    )
                    continue

        # Close any remaining active trade at the end
        if active_trade:
            active_trade.exit_time = prepared.index[-1]
            active_trade.exit_price = prepared.iloc[-1]["close"]
            active_trade.exit_reason = "stop_loss"
            trades.append(active_trade)

        return trades


class RSIMeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy using RSI (Relative Strength Index).
    
    Enters long when RSI drops below oversold threshold (typically <30),
    and enters short when RSI rises above overbought threshold (typically >70).
    Exits when RSI returns to neutral zone or hits stop/target.
    """

    def evaluate(self, frame: pd.DataFrame) -> List[Trade]:
        prepared = self.prepare(frame)
        
        # Add RSI
        relative_strength_index(
            prepared,
            period=self.settings.rsi_period,
            column_name="rsi",
        )
        
        trades: list[Trade] = []
        active_trade: Optional[Trade] = None
        long_triggered = False
        short_triggered = False

        for timestamp, row in prepared.iterrows():
            price = row["close"]
            rsi = row.get("rsi")
            sma_value = row.get("sma")

            # Skip if RSI isn't ready
            if pd.isna(rsi):
                continue

            # Manage active trade
            active_trade, closed_trade = self._manage_active_trade(active_trade, timestamp, row)
            if closed_trade:
                trades.append(closed_trade)
                long_triggered = False
                short_triggered = False
                continue

            # Only enter new trades if we don't have an active one
            if active_trade:
                continue

            # Long entry: RSI oversold (mean reversion bounce)
            if not long_triggered and self.settings.trade_directions in ("long", "both"):
                if rsi <= self.settings.rsi_oversold:
                    # Optional SMA filter: only long if price is above SMA (trend filter)
                    if self.settings.sma_period and (pd.isna(sma_value) or price < sma_value):
                        continue

                    risk_distance = self._compute_risk_distance(
                        "long", price, row.get("atr"), row.get("swing_low"), row.get("swing_high")
                    )
                    if risk_distance is None or risk_distance <= 0:
                        continue

                    quantity = self._position_quantity(price)
                    if quantity is None or quantity <= 0:
                        continue

                    entry_price = price
                    stop_loss = entry_price - risk_distance
                    take_profit = entry_price + self.settings.reward_multiple * risk_distance

                    active_trade = Trade(
                        symbol=self.symbol,
                        direction="long",
                        entry_time=timestamp,
                        entry_price=entry_price,
                        quantity=quantity,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_value=risk_distance,
                        risk_type=self.settings.stop_mode,
                        notes=f"RSI mean reversion: oversold bounce (RSI={rsi:.1f})",
                    )
                    long_triggered = True
                    continue

            # Short entry: RSI overbought (mean reversion fade)
            if not short_triggered and self.settings.trade_directions in ("short", "both"):
                if rsi >= self.settings.rsi_overbought:
                    # Optional SMA filter: only short if price is below SMA (trend filter)
                    if self.settings.sma_period and (pd.isna(sma_value) or price > sma_value):
                        continue

                    risk_distance = self._compute_risk_distance(
                        "short", price, row.get("atr"), row.get("swing_low"), row.get("swing_high")
                    )
                    if risk_distance is None or risk_distance <= 0:
                        continue

                    quantity = self._position_quantity(price)
                    if quantity is None or quantity <= 0:
                        continue

                    entry_price = price
                    stop_loss = entry_price + risk_distance
                    take_profit = entry_price - self.settings.reward_multiple * risk_distance

                    active_trade = Trade(
                        symbol=self.symbol,
                        direction="short",
                        entry_time=timestamp,
                        entry_price=entry_price,
                        quantity=quantity,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_value=risk_distance,
                        risk_type=self.settings.stop_mode,
                        notes=f"RSI mean reversion: overbought fade (RSI={rsi:.1f})",
                    )
                    short_triggered = True
                    continue

        # Close any remaining active trade at the end
        if active_trade:
            active_trade.exit_time = prepared.index[-1]
            active_trade.exit_price = prepared.iloc[-1]["close"]
            active_trade.exit_reason = "stop_loss"
            trades.append(active_trade)

        return trades


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Trend-following strategy using moving average crossovers.
    
    Enters long when fast EMA crosses above slow EMA (golden cross),
    and enters short when fast EMA crosses below slow EMA (death cross).
    """

    def evaluate(self, frame: pd.DataFrame) -> List[Trade]:
        prepared = self.prepare(frame)
        
        # Add EMAs
        exponential_moving_average(
            prepared,
            period=self.settings.ema_fast_period,
            column_name="ema_fast",
        )
        exponential_moving_average(
            prepared,
            period=self.settings.ema_slow_period,
            column_name="ema_slow",
        )
        
        trades: list[Trade] = []
        active_trade: Optional[Trade] = None
        prev_fast: Optional[float] = None
        prev_slow: Optional[float] = None

        for timestamp, row in prepared.iterrows():
            price = row["close"]
            ema_fast = row.get("ema_fast")
            ema_slow = row.get("ema_slow")
            sma_value = row.get("sma")

            # Skip if EMAs aren't ready
            if pd.isna(ema_fast) or pd.isna(ema_slow):
                prev_fast = ema_fast
                prev_slow = ema_slow
                continue

            # Manage active trade
            active_trade, closed_trade = self._manage_active_trade(active_trade, timestamp, row)
            if closed_trade:
                trades.append(closed_trade)
                continue

            # Only enter new trades if we don't have an active one
            if active_trade:
                prev_fast = ema_fast
                prev_slow = ema_slow
                continue

            # Detect crossover
            if prev_fast is not None and prev_slow is not None:
                # Golden cross: fast crosses above slow (bullish)
                if (prev_fast <= prev_slow and ema_fast > ema_slow and 
                    self.settings.trade_directions in ("long", "both")):
                    # Optional SMA filter: only long if price is above SMA
                    if self.settings.sma_period and (pd.isna(sma_value) or price < sma_value):
                        prev_fast = ema_fast
                        prev_slow = ema_slow
                        continue

                    risk_distance = self._compute_risk_distance(
                        "long", price, row.get("atr"), row.get("swing_low"), row.get("swing_high")
                    )
                    if risk_distance is None or risk_distance <= 0:
                        prev_fast = ema_fast
                        prev_slow = ema_slow
                        continue

                    quantity = self._position_quantity(price)
                    if quantity is None or quantity <= 0:
                        prev_fast = ema_fast
                        prev_slow = ema_slow
                        continue

                    entry_price = price
                    stop_loss = entry_price - risk_distance
                    take_profit = entry_price + self.settings.reward_multiple * risk_distance

                    active_trade = Trade(
                        symbol=self.symbol,
                        direction="long",
                        entry_time=timestamp,
                        entry_price=entry_price,
                        quantity=quantity,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_value=risk_distance,
                        risk_type=self.settings.stop_mode,
                        notes=f"EMA crossover: golden cross (fast={ema_fast:.2f}, slow={ema_slow:.2f})",
                    )
                    prev_fast = ema_fast
                    prev_slow = ema_slow
                    continue

                # Death cross: fast crosses below slow (bearish)
                if (prev_fast >= prev_slow and ema_fast < ema_slow and 
                    self.settings.trade_directions in ("short", "both")):
                    # Optional SMA filter: only short if price is below SMA
                    if self.settings.sma_period and (pd.isna(sma_value) or price > sma_value):
                        prev_fast = ema_fast
                        prev_slow = ema_slow
                        continue

                    risk_distance = self._compute_risk_distance(
                        "short", price, row.get("atr"), row.get("swing_low"), row.get("swing_high")
                    )
                    if risk_distance is None or risk_distance <= 0:
                        prev_fast = ema_fast
                        prev_slow = ema_slow
                        continue

                    quantity = self._position_quantity(price)
                    if quantity is None or quantity <= 0:
                        prev_fast = ema_fast
                        prev_slow = ema_slow
                        continue

                    entry_price = price
                    stop_loss = entry_price + risk_distance
                    take_profit = entry_price - self.settings.reward_multiple * risk_distance

                    active_trade = Trade(
                        symbol=self.symbol,
                        direction="short",
                        entry_time=timestamp,
                        entry_price=entry_price,
                        quantity=quantity,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_value=risk_distance,
                        risk_type=self.settings.stop_mode,
                        notes=f"EMA crossover: death cross (fast={ema_fast:.2f}, slow={ema_slow:.2f})",
                    )
                    prev_fast = ema_fast
                    prev_slow = ema_slow
                    continue

            prev_fast = ema_fast
            prev_slow = ema_slow

        # Close any remaining active trade at the end
        if active_trade:
            active_trade.exit_time = prepared.index[-1]
            active_trade.exit_price = prepared.iloc[-1]["close"]
            active_trade.exit_reason = "stop_loss"
            trades.append(active_trade)

        return trades


class VolumeBreakoutStrategy(BaseStrategy):
    """
    Breakout strategy with volume confirmation.
    
    Enters long when price breaks above recent high with volume above average,
    and enters short when price breaks below recent low with volume above average.
    Uses volume spikes to confirm breakouts.
    """

    def evaluate(self, frame: pd.DataFrame) -> List[Trade]:
        prepared = self.prepare(frame)
        
        # Add volume moving average
        if "volume" not in prepared.columns:
            # If no volume data, skip this strategy
            return []
        
        volume_moving_average(
            prepared,
            period=self.settings.volume_ma_period,
            column_name="volume_ma",
        )
        
        trades: list[Trade] = []
        active_trade: Optional[Trade] = None
        
        # Track recent highs and lows for breakout detection
        lookback_period = max(20, self.settings.volume_ma_period)
        prepared["recent_high"] = prepared["high"].rolling(window=lookback_period, min_periods=lookback_period).max()
        prepared["recent_low"] = prepared["low"].rolling(window=lookback_period, min_periods=lookback_period).min()

        for timestamp, row in prepared.iterrows():
            price = row["close"]
            volume = row.get("volume", 0)
            volume_ma = row.get("volume_ma")
            recent_high = row.get("recent_high")
            recent_low = row.get("recent_low")
            sma_value = row.get("sma")

            # Skip if indicators aren't ready
            if pd.isna(volume_ma) or pd.isna(recent_high) or pd.isna(recent_low):
                continue

            # Manage active trade
            active_trade, closed_trade = self._manage_active_trade(active_trade, timestamp, row)
            if closed_trade:
                trades.append(closed_trade)
                continue

            # Only enter new trades if we don't have an active one
            if active_trade:
                continue

            # Long entry: price breaks above recent high with volume confirmation
            if self.settings.trade_directions in ("long", "both"):
                if row["high"] > recent_high and volume >= volume_ma * self.settings.volume_multiplier:
                    # Optional SMA filter: only long if price is above SMA
                    if self.settings.sma_period and (pd.isna(sma_value) or price < sma_value):
                        continue

                    risk_distance = self._compute_risk_distance(
                        "long", price, row.get("atr"), row.get("swing_low"), row.get("swing_high")
                    )
                    if risk_distance is None or risk_distance <= 0:
                        continue

                    quantity = self._position_quantity(price)
                    if quantity is None or quantity <= 0:
                        continue

                    entry_price = price
                    stop_loss = entry_price - risk_distance
                    take_profit = entry_price + self.settings.reward_multiple * risk_distance

                    active_trade = Trade(
                        symbol=self.symbol,
                        direction="long",
                        entry_time=timestamp,
                        entry_price=entry_price,
                        quantity=quantity,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_value=risk_distance,
                        risk_type=self.settings.stop_mode,
                        notes=f"Volume breakout: above {recent_high:.2f} (vol={volume:.0f}, avg={volume_ma:.0f})",
                    )
                    continue

            # Short entry: price breaks below recent low with volume confirmation
            if self.settings.trade_directions in ("short", "both"):
                if row["low"] < recent_low and volume >= volume_ma * self.settings.volume_multiplier:
                    # Optional SMA filter: only short if price is below SMA
                    if self.settings.sma_period and (pd.isna(sma_value) or price > sma_value):
                        continue

                    risk_distance = self._compute_risk_distance(
                        "short", price, row.get("atr"), row.get("swing_low"), row.get("swing_high")
                    )
                    if risk_distance is None or risk_distance <= 0:
                        continue

                    quantity = self._position_quantity(price)
                    if quantity is None or quantity <= 0:
                        continue

                    entry_price = price
                    stop_loss = entry_price + risk_distance
                    take_profit = entry_price - self.settings.reward_multiple * risk_distance

                    active_trade = Trade(
                        symbol=self.symbol,
                        direction="short",
                        entry_time=timestamp,
                        entry_price=entry_price,
                        quantity=quantity,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_value=risk_distance,
                        risk_type=self.settings.stop_mode,
                        notes=f"Volume breakout: below {recent_low:.2f} (vol={volume:.0f}, avg={volume_ma:.0f})",
                    )
                    continue

        # Close any remaining active trade at the end
        if active_trade:
            active_trade.exit_time = prepared.index[-1]
            active_trade.exit_price = prepared.iloc[-1]["close"]
            active_trade.exit_reason = "stop_loss"
            trades.append(active_trade)

        return trades


class StochasticOscillatorStrategy(BaseStrategy):
    """
    Mean reversion strategy using Stochastic Oscillator.
    
    Enters long when Stochastic %K drops below oversold threshold (typically <20),
    and enters short when Stochastic %K rises above overbought threshold (typically >80).
    Uses %D (smoothed %K) for confirmation.
    """

    def evaluate(self, frame: pd.DataFrame) -> List[Trade]:
        prepared = self.prepare(frame)
        
        # Add Stochastic Oscillator
        stochastic_oscillator(
            prepared,
            k_period=self.settings.stoch_k_period,
            d_period=self.settings.stoch_d_period,
            k_column="stoch_k",
            d_column="stoch_d",
        )
        
        trades: list[Trade] = []
        active_trade: Optional[Trade] = None
        long_triggered = False
        short_triggered = False

        for timestamp, row in prepared.iterrows():
            price = row["close"]
            stoch_k = row.get("stoch_k")
            stoch_d = row.get("stoch_d")
            sma_value = row.get("sma")

            # Skip if Stochastic isn't ready
            if pd.isna(stoch_k) or pd.isna(stoch_d):
                continue

            # Manage active trade
            active_trade, closed_trade = self._manage_active_trade(active_trade, timestamp, row)
            if closed_trade:
                trades.append(closed_trade)
                long_triggered = False
                short_triggered = False
                continue

            # Only enter new trades if we don't have an active one
            if active_trade:
                continue

            # Long entry: Stochastic oversold (mean reversion bounce)
            if not long_triggered and self.settings.trade_directions in ("long", "both"):
                if stoch_k <= self.settings.stoch_oversold and stoch_d <= self.settings.stoch_oversold:
                    # Optional SMA filter: only long if price is above SMA (trend filter)
                    if self.settings.sma_period and (pd.isna(sma_value) or price < sma_value):
                        continue

                    risk_distance = self._compute_risk_distance(
                        "long", price, row.get("atr"), row.get("swing_low"), row.get("swing_high")
                    )
                    if risk_distance is None or risk_distance <= 0:
                        continue

                    quantity = self._position_quantity(price)
                    if quantity is None or quantity <= 0:
                        continue

                    entry_price = price
                    stop_loss = entry_price - risk_distance
                    take_profit = entry_price + self.settings.reward_multiple * risk_distance

                    active_trade = Trade(
                        symbol=self.symbol,
                        direction="long",
                        entry_time=timestamp,
                        entry_price=entry_price,
                        quantity=quantity,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_value=risk_distance,
                        risk_type=self.settings.stop_mode,
                        notes=f"Stochastic oversold: %K={stoch_k:.1f}, %D={stoch_d:.1f}",
                    )
                    long_triggered = True
                    continue

            # Short entry: Stochastic overbought (mean reversion fade)
            if not short_triggered and self.settings.trade_directions in ("short", "both"):
                if stoch_k >= self.settings.stoch_overbought and stoch_d >= self.settings.stoch_overbought:
                    # Optional SMA filter: only short if price is below SMA (trend filter)
                    if self.settings.sma_period and (pd.isna(sma_value) or price > sma_value):
                        continue

                    risk_distance = self._compute_risk_distance(
                        "short", price, row.get("atr"), row.get("swing_low"), row.get("swing_high")
                    )
                    if risk_distance is None or risk_distance <= 0:
                        continue

                    quantity = self._position_quantity(price)
                    if quantity is None or quantity <= 0:
                        continue

                    entry_price = price
                    stop_loss = entry_price + risk_distance
                    take_profit = entry_price - self.settings.reward_multiple * risk_distance

                    active_trade = Trade(
                        symbol=self.symbol,
                        direction="short",
                        entry_time=timestamp,
                        entry_price=entry_price,
                        quantity=quantity,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_value=risk_distance,
                        risk_type=self.settings.stop_mode,
                        notes=f"Stochastic overbought: %K={stoch_k:.1f}, %D={stoch_d:.1f}",
                    )
                    short_triggered = True
                    continue

        # Close any remaining active trade at the end
        if active_trade:
            active_trade.exit_time = prepared.index[-1]
            active_trade.exit_price = prepared.iloc[-1]["close"]
            active_trade.exit_reason = "stop_loss"
            trades.append(active_trade)

        return trades


class MACDCrossoverStrategy(BaseStrategy):
    """
    Trend-following strategy using MACD signal line crossovers.
    
    Enters long when MACD line crosses above signal line (bullish crossover),
    and enters short when MACD line crosses below signal line (bearish crossover).
    """

    def evaluate(self, frame: pd.DataFrame) -> List[Trade]:
        prepared = self.prepare(frame)
        
        # Add MACD
        macd(
            prepared,
            fast_period=self.settings.macd_fast_period,
            slow_period=self.settings.macd_slow_period,
            signal_period=self.settings.macd_signal_period,
            macd_column="macd",
            signal_column="macd_signal",
            histogram_column="macd_histogram",
        )
        
        trades: list[Trade] = []
        active_trade: Optional[Trade] = None
        prev_macd: Optional[float] = None
        prev_signal: Optional[float] = None

        for timestamp, row in prepared.iterrows():
            price = row["close"]
            macd_line = row.get("macd")
            macd_signal = row.get("macd_signal")
            sma_value = row.get("sma")

            # Skip if MACD isn't ready
            if pd.isna(macd_line) or pd.isna(macd_signal):
                prev_macd = macd_line
                prev_signal = macd_signal
                continue

            # Manage active trade
            active_trade, closed_trade = self._manage_active_trade(active_trade, timestamp, row)
            if closed_trade:
                trades.append(closed_trade)
                continue

            # Only enter new trades if we don't have an active one
            if active_trade:
                prev_macd = macd_line
                prev_signal = macd_signal
                continue

            # Detect crossover
            if prev_macd is not None and prev_signal is not None:
                # Bullish crossover: MACD crosses above signal
                if (prev_macd <= prev_signal and macd_line > macd_signal and 
                    self.settings.trade_directions in ("long", "both")):
                    # Optional SMA filter: only long if price is above SMA
                    if self.settings.sma_period and (pd.isna(sma_value) or price < sma_value):
                        prev_macd = macd_line
                        prev_signal = macd_signal
                        continue

                    risk_distance = self._compute_risk_distance(
                        "long", price, row.get("atr"), row.get("swing_low"), row.get("swing_high")
                    )
                    if risk_distance is None or risk_distance <= 0:
                        prev_macd = macd_line
                        prev_signal = macd_signal
                        continue

                    quantity = self._position_quantity(price)
                    if quantity is None or quantity <= 0:
                        prev_macd = macd_line
                        prev_signal = macd_signal
                        continue

                    entry_price = price
                    stop_loss = entry_price - risk_distance
                    take_profit = entry_price + self.settings.reward_multiple * risk_distance

                    active_trade = Trade(
                        symbol=self.symbol,
                        direction="long",
                        entry_time=timestamp,
                        entry_price=entry_price,
                        quantity=quantity,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_value=risk_distance,
                        risk_type=self.settings.stop_mode,
                        notes=f"MACD bullish crossover: MACD={macd_line:.2f}, Signal={macd_signal:.2f}",
                    )
                    prev_macd = macd_line
                    prev_signal = macd_signal
                    continue

                # Bearish crossover: MACD crosses below signal
                if (prev_macd >= prev_signal and macd_line < macd_signal and 
                    self.settings.trade_directions in ("short", "both")):
                    # Optional SMA filter: only short if price is below SMA
                    if self.settings.sma_period and (pd.isna(sma_value) or price > sma_value):
                        prev_macd = macd_line
                        prev_signal = macd_signal
                        continue

                    risk_distance = self._compute_risk_distance(
                        "short", price, row.get("atr"), row.get("swing_low"), row.get("swing_high")
                    )
                    if risk_distance is None or risk_distance <= 0:
                        prev_macd = macd_line
                        prev_signal = macd_signal
                        continue

                    quantity = self._position_quantity(price)
                    if quantity is None or quantity <= 0:
                        prev_macd = macd_line
                        prev_signal = macd_signal
                        continue

                    entry_price = price
                    stop_loss = entry_price + risk_distance
                    take_profit = entry_price - self.settings.reward_multiple * risk_distance

                    active_trade = Trade(
                        symbol=self.symbol,
                        direction="short",
                        entry_time=timestamp,
                        entry_price=entry_price,
                        quantity=quantity,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_value=risk_distance,
                        risk_type=self.settings.stop_mode,
                        notes=f"MACD bearish crossover: MACD={macd_line:.2f}, Signal={macd_signal:.2f}",
                    )
                    prev_macd = macd_line
                    prev_signal = macd_signal
                    continue

            prev_macd = macd_line
            prev_signal = macd_signal

        # Close any remaining active trade at the end
        if active_trade:
            active_trade.exit_time = prepared.index[-1]
            active_trade.exit_price = prepared.iloc[-1]["close"]
            active_trade.exit_reason = "stop_loss"
            trades.append(active_trade)

        return trades


class ADXTrendStrengthStrategy(BaseStrategy):
    """
    Trend-following strategy using ADX (Average Directional Index) for trend strength confirmation.
    
    Enters long when price is above EMA and ADX shows strong trend (ADX > threshold),
    and enters short when price is below EMA and ADX shows strong trend.
    Only trades when trend strength is confirmed by ADX.
    """

    def evaluate(self, frame: pd.DataFrame) -> List[Trade]:
        prepared = self.prepare(frame)
        
        # Add ADX and EMA for trend direction
        average_directional_index(
            prepared,
            period=self.settings.adx_period,
            column_name="adx",
        )
        exponential_moving_average(
            prepared,
            period=self.settings.ema_slow_period,
            column_name="ema_trend",
        )
        
        trades: list[Trade] = []
        active_trade: Optional[Trade] = None

        for timestamp, row in prepared.iterrows():
            price = row["close"]
            adx = row.get("adx")
            ema_trend = row.get("ema_trend")
            sma_value = row.get("sma")

            # Skip if indicators aren't ready
            if pd.isna(adx) or pd.isna(ema_trend):
                continue

            # Manage active trade
            active_trade, closed_trade = self._manage_active_trade(active_trade, timestamp, row)
            if closed_trade:
                trades.append(closed_trade)
                continue

            # Only enter new trades if we don't have an active one
            if active_trade:
                continue

            # Only trade when ADX shows strong trend
            if adx < self.settings.adx_threshold:
                continue

            # Long entry: price above EMA with strong ADX (uptrend)
            if self.settings.trade_directions in ("long", "both"):
                if price > ema_trend:
                    # Optional SMA filter: only long if price is above SMA
                    if self.settings.sma_period and (pd.isna(sma_value) or price < sma_value):
                        continue

                    risk_distance = self._compute_risk_distance(
                        "long", price, row.get("atr"), row.get("swing_low"), row.get("swing_high")
                    )
                    if risk_distance is None or risk_distance <= 0:
                        continue

                    quantity = self._position_quantity(price)
                    if quantity is None or quantity <= 0:
                        continue

                    entry_price = price
                    stop_loss = entry_price - risk_distance
                    take_profit = entry_price + self.settings.reward_multiple * risk_distance

                    active_trade = Trade(
                        symbol=self.symbol,
                        direction="long",
                        entry_time=timestamp,
                        entry_price=entry_price,
                        quantity=quantity,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_value=risk_distance,
                        risk_type=self.settings.stop_mode,
                        notes=f"ADX trend strength: ADX={adx:.1f}, price above EMA {ema_trend:.2f}",
                    )
                    continue

            # Short entry: price below EMA with strong ADX (downtrend)
            if self.settings.trade_directions in ("short", "both"):
                if price < ema_trend:
                    # Optional SMA filter: only short if price is below SMA
                    if self.settings.sma_period and (pd.isna(sma_value) or price > sma_value):
                        continue

                    risk_distance = self._compute_risk_distance(
                        "short", price, row.get("atr"), row.get("swing_low"), row.get("swing_high")
                    )
                    if risk_distance is None or risk_distance <= 0:
                        continue

                    quantity = self._position_quantity(price)
                    if quantity is None or quantity <= 0:
                        continue

                    entry_price = price
                    stop_loss = entry_price + risk_distance
                    take_profit = entry_price - self.settings.reward_multiple * risk_distance

                    active_trade = Trade(
                        symbol=self.symbol,
                        direction="short",
                        entry_time=timestamp,
                        entry_price=entry_price,
                        quantity=quantity,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_value=risk_distance,
                        risk_type=self.settings.stop_mode,
                        notes=f"ADX trend strength: ADX={adx:.1f}, price below EMA {ema_trend:.2f}",
                    )
                    continue

        # Close any remaining active trade at the end
        if active_trade:
            active_trade.exit_time = prepared.index[-1]
            active_trade.exit_price = prepared.iloc[-1]["close"]
            active_trade.exit_reason = "stop_loss"
            trades.append(active_trade)

        return trades

