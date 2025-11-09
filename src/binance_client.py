"""
Utilities for creating a Binance client via ccxt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import ccxt  # type: ignore


@dataclass
class BinanceConfig:
    """Configuration for connecting to Binance via ccxt."""

    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    use_testnet: bool = False


def create_client(config: BinanceConfig) -> ccxt.binance:
    """
    Create a configured ccxt Binance client.

    Parameters
    ----------
    config:
        The configuration with credentials and environment selection.
    """

    exchange = ccxt.binance(
        {
            "apiKey": config.api_key,
            "secret": config.api_secret,
            "enableRateLimit": True,
            "options": {
                # Ensure all timestamps are strictly UTC
                "defaultType": "future" if config.use_testnet else "spot",
            },
        }
    )

    if config.use_testnet:
        exchange.set_sandbox_mode(True)

    return exchange

