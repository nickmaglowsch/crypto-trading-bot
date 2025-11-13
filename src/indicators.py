"""
Technical indicator utilities using pandas-ta library for optimized performance.
"""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta


def average_true_range(
    frame: pd.DataFrame,
    *,
    period: int = 14,
    column_name: str = "atr",
) -> pd.Series:
    """
    Compute the Average True Range (ATR) using pandas-ta.

    Parameters
    ----------
    frame:
        DataFrame with ``high``, ``low`` and ``close`` columns.
    period:
        ATR lookback period.
    column_name:
        Optional column name for assigning ATR to the frame.
    """
    atr = frame.ta.atr(length=period)
    
    if column_name and atr is not None:
        frame[column_name] = atr

    return atr if atr is not None else pd.Series(dtype=float, index=frame.index)


def simple_moving_average(
    frame: pd.DataFrame,
    *,
    period: int,
    column_name: str = "sma",
) -> pd.Series:
    """
    Compute a simple moving average of the close price using pandas-ta.

    Parameters
    ----------
    frame:
        DataFrame with a ``close`` column.
    period:
        SMA lookback period.
    column_name:
        Optional column name for assigning SMA to the frame.
    """
    sma = frame.ta.sma(length=period)
    
    if column_name and sma is not None:
        frame[column_name] = sma

    return sma if sma is not None else pd.Series(dtype=float, index=frame.index)


def bollinger_bands(
    frame: pd.DataFrame,
    *,
    period: int = 20,
    num_std: float = 2.0,
    sma_column: str = "sma",
    upper_column: str = "bb_upper",
    lower_column: str = "bb_lower",
    middle_column: str = "bb_middle",
) -> pd.DataFrame:
    """
    Compute Bollinger Bands (upper, middle, lower) using pandas-ta.

    Parameters
    ----------
    frame:
        DataFrame with a ``close`` column.
    period:
        Lookback period for the moving average and standard deviation.
    num_std:
        Number of standard deviations for the bands.
    sma_column:
        Column name for the middle band (SMA) - kept for compatibility.
    upper_column:
        Column name for the upper band.
    lower_column:
        Column name for the lower band.
    middle_column:
        Column name for the middle band (same as SMA).
    """
    bb = frame.ta.bbands(length=period, std=num_std)
    
    if bb is not None and not bb.empty:
        # pandas-ta returns DataFrame with columns like 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0'
        # Find columns by name pattern
        bb_cols = bb.columns.tolist()
        lower_col = next((c for c in bb_cols if c.startswith('BBL')), None)
        middle_col = next((c for c in bb_cols if c.startswith('BBM')), None)
        upper_col = next((c for c in bb_cols if c.startswith('BBU')), None)
        
        if lower_col and middle_col and upper_col:
            frame[lower_column] = bb[lower_col]
            frame[middle_column] = bb[middle_col]
            frame[upper_column] = bb[upper_col]
            # Also set sma_column for compatibility
            if sma_column:
                frame[sma_column] = bb[middle_col]
        elif len(bb_cols) >= 3:
            # Fallback: use column order if pattern matching fails
            frame[lower_column] = bb.iloc[:, 0]
            frame[middle_column] = bb.iloc[:, 1]
            frame[upper_column] = bb.iloc[:, 2]
            if sma_column:
                frame[sma_column] = bb.iloc[:, 1]

    return frame


def relative_strength_index(
    frame: pd.DataFrame,
    *,
    period: int = 14,
    column_name: str = "rsi",
) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) using pandas-ta.

    Parameters
    ----------
    frame:
        DataFrame with a ``close`` column.
    period:
        RSI lookback period.
    column_name:
        Column name for assigning RSI to the frame.
    """
    rsi = frame.ta.rsi(length=period)
    
    if column_name and rsi is not None:
        frame[column_name] = rsi
    
    return rsi if rsi is not None else pd.Series(dtype=float, index=frame.index)


def exponential_moving_average(
    frame: pd.DataFrame,
    *,
    period: int,
    column_name: str = "ema",
) -> pd.Series:
    """
    Compute an exponential moving average of the close price using pandas-ta.

    Parameters
    ----------
    frame:
        DataFrame with a ``close`` column.
    period:
        EMA lookback period.
    column_name:
        Column name for assigning EMA to the frame.
    """
    ema = frame.ta.ema(length=period)
    
    if column_name and ema is not None:
        frame[column_name] = ema
    
    return ema if ema is not None else pd.Series(dtype=float, index=frame.index)


def macd(
    frame: pd.DataFrame,
    *,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    macd_column: str = "macd",
    signal_column: str = "macd_signal",
    histogram_column: str = "macd_histogram",
) -> pd.DataFrame:
    """
    Compute MACD (Moving Average Convergence Divergence) using pandas-ta.

    Parameters
    ----------
    frame:
        DataFrame with a ``close`` column.
    fast_period:
        Fast EMA period.
    slow_period:
        Slow EMA period.
    signal_period:
        Signal line EMA period.
    macd_column:
        Column name for MACD line.
    signal_column:
        Column name for signal line.
    histogram_column:
        Column name for MACD histogram (MACD - Signal).
    """
    macd_df = frame.ta.macd(fast=fast_period, slow=slow_period, signal=signal_period)
    
    if macd_df is not None and not macd_df.empty:
        # pandas-ta returns DataFrame with columns like 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9'
        # Find columns by name pattern
        macd_cols = macd_df.columns.tolist()
        macd_line_col = next((c for c in macd_cols if c.startswith('MACD_') and not c.startswith('MACDs') and not c.startswith('MACDh')), None)
        signal_col = next((c for c in macd_cols if c.startswith('MACDs')), None)
        hist_col = next((c for c in macd_cols if c.startswith('MACDh')), None)
        
        if macd_line_col and signal_col and hist_col:
            frame[macd_column] = macd_df[macd_line_col]
            frame[signal_column] = macd_df[signal_col]
            frame[histogram_column] = macd_df[hist_col]
        elif len(macd_cols) >= 3:
            # Fallback: use column order if pattern matching fails
            frame[macd_column] = macd_df.iloc[:, 0]
            frame[signal_column] = macd_df.iloc[:, 1]
            frame[histogram_column] = macd_df.iloc[:, 2]
    
    return frame


def volume_moving_average(
    frame: pd.DataFrame,
    *,
    period: int = 20,
    column_name: str = "volume_ma",
) -> pd.Series:
    """
    Compute a simple moving average of volume using pandas-ta.

    Parameters
    ----------
    frame:
        DataFrame with a ``volume`` column.
    period:
        Volume MA lookback period.
    column_name:
        Column name for assigning volume MA to the frame.
    """
    if "volume" not in frame.columns:
        raise ValueError("DataFrame must have a 'volume' column")
    
    # Create temporary DataFrame with volume as 'close' for pandas-ta
    temp_df = pd.DataFrame(index=frame.index)
    temp_df["close"] = frame["volume"]
    volume_ma = temp_df.ta.sma(length=period)
    
    if column_name and volume_ma is not None:
        frame[column_name] = volume_ma
    
    return volume_ma if volume_ma is not None else pd.Series(dtype=float, index=frame.index)