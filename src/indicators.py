"""
Technical indicator utilities.
"""

from __future__ import annotations

import pandas as pd


def average_true_range(
    frame: pd.DataFrame,
    *,
    period: int = 14,
    column_name: str = "atr",
) -> pd.Series:
    """
    Compute the Average True Range (ATR).

    Parameters
    ----------
    frame:
        DataFrame with ``high``, ``low`` and ``close`` columns.
    period:
        ATR lookback period.
    column_name:
        Optional column name for assigning ATR to the frame.
    """

    high = frame["high"]
    low = frame["low"]
    close = frame["close"]

    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=period).mean()

    if column_name:
        frame[column_name] = atr

    return atr


def simple_moving_average(
    frame: pd.DataFrame,
    *,
    period: int,
    column_name: str = "sma",
) -> pd.Series:
    """
    Compute a simple moving average of the close price.

    Parameters
    ----------
    frame:
        DataFrame with a ``close`` column.
    period:
        SMA lookback period.
    column_name:
        Optional column name for assigning SMA to the frame.
    """

    sma = frame["close"].rolling(window=period, min_periods=period).mean()

    if column_name:
        frame[column_name] = sma

    return sma

