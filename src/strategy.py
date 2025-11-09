"""
Strategy framework with shared trade models and concrete implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import List, Literal, Optional, Tuple

import pandas as pd

from .indicators import average_true_range, simple_moving_average

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

    def pnl(self) -> Optional[float]:
        """Return absolute profit or loss if the trade has closed."""
        if self.exit_price is None or self.exit_reason is None:
            return None

        if self.direction == "long":
            raw = (self.exit_price - self.entry_price) * self.quantity
        else:
            raw = (self.entry_price - self.exit_price) * self.quantity

        return raw * self.cost_multiplier


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

