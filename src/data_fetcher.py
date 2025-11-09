"""
Helpers to load OHLCV data from Binance.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from .binance_client import BinanceConfig, create_client


def fetch_ohlcv_dataframe(
    symbol: str,
    timeframe: str,
    *,
    days: int = 1,
    end_time: Optional[datetime] = None,
    config: Optional[BinanceConfig] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    cache_ttl: Optional[timedelta] = None,
) -> pd.DataFrame:
    """
    Fetch recent OHLCV candles and return them as a pandas DataFrame.

    Parameters
    ----------
    symbol:
        The trading pair symbol, e.g. ``"BTC/USDT"``.
    timeframe:
        The candle timeframe supported by Binance, e.g. ``"5m"`` or ``"1m"``.
    days:
        Number of trailing days to load. Defaults to the most recent day.
    end_time:
        Optional end timestamp (UTC). Defaults to current time.
    config:
        Optional configuration for authentication / testnet.
    """

    if end_time is None:
        end_time = datetime.now(tz=timezone.utc)
    else:
        end_time = end_time.astimezone(timezone.utc)

    start_time = end_time - timedelta(days=days)

    cache_path: Optional[Path] = None
    if cache_dir:
        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        safe_symbol = symbol.replace("/", "_")
        cache_filename = f"{safe_symbol}_{timeframe}_{days}.pkl"
        cache_path = cache_dir_path / cache_filename

        if cache_path.exists():
            cache_age = datetime.now(tz=timezone.utc) - datetime.fromtimestamp(
                cache_path.stat().st_mtime, tz=timezone.utc
            )
            if cache_ttl is None or cache_age <= cache_ttl:
                return pd.read_pickle(cache_path)

    client = create_client(config or BinanceConfig())

    since_ms = int(start_time.timestamp() * 1000)

    # Limit ~1500 candles per request, so fetch in batches if needed.
    ohlcv_rows: list[Sequence[float]] = []
    cursor = since_ms
    while True:
        batch = client.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1500)
        if not batch:
            break

        ohlcv_rows.extend(batch)
        cursor = int(batch[-1][0]) + 1

        if batch[-1][0] >= int(end_time.timestamp() * 1000):
            break

    if not ohlcv_rows:
        raise RuntimeError("No OHLCV data returned from Binance.")

    frame = pd.DataFrame(
        ohlcv_rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame.set_index("timestamp", inplace=True)
    frame = frame.astype(
        {
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        }
    )

    if cache_path:
        frame.to_pickle(cache_path)

    return frame

