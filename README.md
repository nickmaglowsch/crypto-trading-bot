# Algo Trade – Strategy Sandbox

This project provides a small framework for experimenting with Binance spot strategies. It currently includes:

**Mean Reversion Strategies:**
- **Intraday Range Fade** – watch the first 4 hours of the session, fade breaks that return into the range.
- **Bollinger Bands Mean Reversion** – trade bounces off the upper/lower Bollinger Bands, targeting mean reversion back to the middle band.
- **RSI Mean Reversion** – trade oversold/overbought conditions when RSI drops below 30 or rises above 70.
- **Stochastic Oscillator** – trade oversold/overbought conditions when Stochastic %K and %D drop below 20 or rise above 80.

**Trend Following Strategies:**
- **Momentum Breakout** – go with momentum when price pushes above yesterday's high or below yesterday's low.
- **Moving Average Crossover** – trade golden cross (fast EMA crosses above slow EMA) and death cross (fast EMA crosses below slow EMA).
- **Volume Breakout** – trade breakouts above recent highs or below recent lows, confirmed by volume spikes above average.
- **MACD Crossover** – trade when MACD line crosses above/below signal line (bullish/bearish crossovers).
- **ADX Trend Strength** – trade strong trends confirmed by ADX (only enters when ADX > threshold and price aligns with EMA direction).

The code uses [`ccxt`](https://github.com/ccxt/ccxt) to pull historical candles and pandas to backtest the logic.

## Getting Started

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the strategy on recent data

```bash
python -m src.run_strategy --symbol BTC/USDT --timeframe 5m --days 7
```

To try the momentum breakout with swing-based risk:

```bash
python -m src.run_strategy --strategy momentum --symbol BTC/USDT --timeframe 15m --days 10 --stop-mode swing --swing-period 20
```

To try the Bollinger Bands mean reversion strategy:

```bash
python -m src.run_strategy --strategy bollinger_reversion --symbol BTC/USDT --timeframe 15m --days 10 --bb-period 20 --bb-std 2.0
```

To try the RSI mean reversion strategy:

```bash
python -m src.run_strategy --strategy rsi_reversion --symbol BTC/USDT --timeframe 15m --days 10 --rsi-period 14 --rsi-oversold 30 --rsi-overbought 70
```

To try the Moving Average Crossover strategy:

```bash
python -m src.run_strategy --strategy ma_crossover --symbol BTC/USDT --timeframe 15m --days 10 --ema-fast 12 --ema-slow 26
```

To try the Volume Breakout strategy:

```bash
python -m src.run_strategy --strategy volume_breakout --symbol BTC/USDT --timeframe 15m --days 10 --volume-ma-period 20 --volume-multiplier 1.5
```

To try the Stochastic Oscillator strategy:

```bash
python -m src.run_strategy --strategy stochastic --symbol BTC/USDT --timeframe 15m --days 10 --stoch-k-period 14 --stoch-d-period 3
```

To try the MACD Crossover strategy:

```bash
python -m src.run_strategy --strategy macd_crossover --symbol BTC/USDT --timeframe 15m --days 10
```

To try the ADX Trend Strength strategy:

```bash
python -m src.run_strategy --strategy adx_trend --symbol BTC/USDT --timeframe 15m --days 10 --adx-threshold 25.0
```

Command line options:

- `--strategy`: Choose `intraday_range` (default), `momentum`, `bollinger_reversion`, `rsi_reversion`, `ma_crossover`, `volume_breakout`, `stochastic`, `macd_crossover`, or `adx_trend`.
- `--symbol`: Binance market symbol or comma-separated list for a multi-asset run.
- `--timeframe`: Candle timeframe (1m, 5m, 15m, etc).
- `--days`: Number of trailing days to evaluate.
- `--atr-period`: Lookback for ATR (default 14).
- `--reward-multiple`: Take-profit multiplier vs ATR (default 2.0).
- `--initial-capital`: Starting balance for the backtest equity curve (default 10,000).
- `--session-offset`: Hour offset from UTC that defines the start of the trading day (e.g. `-3` for UTC-3).
- `--cache-dir`: Location to store cached OHLCV data (default `.cache`; set empty to disable caching).
- `--cache-ttl-hours`: Reuse cached data younger than this many hours (default 6.0, set 0 to force refresh).
- `--plot-equity`: Show a matplotlib equity curve after the run (requires `matplotlib`).
- `--position-fraction`: Fraction of capital used per trade (1.0 = 100%, >1.0 allows leverage).
- `--trading-cost`: Percentage trading cost deducted per trade (e.g. 0.001 = 0.1%, applied on both entry and exit).
- `--sma-period`: Apply a simple moving average filter (longs only above SMA, shorts only below).
- `--stop-mode`: Choose `atr` (default) or `swing` to use distance from the latest swing high/low as the stop.
- `--swing-period`: Number of candles to look back for swing-based stops (defaults to 20 when `--stop-mode swing`).
- `--trade-directions`: Limit trades to `long`, `short`, or `both` (default).
- `--bb-period`: Bollinger Bands period (default 20, only for `bollinger_reversion` strategy).
- `--bb-std`: Bollinger Bands standard deviation multiplier (default 2.0, only for `bollinger_reversion` strategy).
- `--rsi-period`: RSI lookback period (default 14, only for `rsi_reversion` strategy).
- `--rsi-oversold`: RSI oversold threshold for long entries (default 30.0, only for `rsi_reversion` strategy).
- `--rsi-overbought`: RSI overbought threshold for short entries (default 70.0, only for `rsi_reversion` strategy).
- `--ema-fast`: Fast EMA period (default 12, only for `ma_crossover` strategy).
- `--ema-slow`: Slow EMA period (default 26, only for `ma_crossover` strategy).
- `--volume-ma-period`: Volume moving average period (default 20, only for `volume_breakout` strategy).
- `--volume-multiplier`: Volume multiplier threshold for breakout confirmation (default 1.5, only for `volume_breakout` strategy).
- `--stoch-k-period`: Stochastic %K period (default 14, only for `stochastic` strategy).
- `--stoch-d-period`: Stochastic %D smoothing period (default 3, only for `stochastic` strategy).
- `--stoch-oversold`: Stochastic oversold threshold for long entries (default 20.0, only for `stochastic` strategy).
- `--stoch-overbought`: Stochastic overbought threshold for short entries (default 80.0, only for `stochastic` strategy).
- `--adx-period`: ADX lookback period (default 14, only for `adx_trend` strategy).
- `--adx-threshold`: ADX threshold for trend strength confirmation (default 25.0, only for `adx_trend` strategy).
- `--testnet`: Toggle Binance sandbox mode for future live order routing.
- `--api-key`/`--api-secret`: Optional credentials (not needed for historical data).

### 3. Output

The script prints each trade, its entry/exit details and cumulative PnL (in quote currency units). Use this to gauge performance before considering a live deployment.

The summary now includes basic backtest metrics such as win rate, average win/loss, expectancy, an equity curve based on the chosen starting capital, plus separate long/short win rates and average outcomes. When you pass multiple symbols, the runner evaluates each asset individually and combines trades into a portfolio while respecting the maximum simultaneous holdings implied by `--position-fraction` (e.g. `0.5` allows up to two concurrent positions).

Additional analytics printed after each run:

- Sharpe ratio of the combined portfolio.
- Portfolio-level maximum drawdown.
- Per-asset maximum drawdown.
- Average holding time (in minutes) for each asset.
- Total fees paid across all trades.

## Next Steps

- Connect the signals to live order execution via ccxt.
- Add position sizing and risk controls.
- Expand with unit tests and a more robust back-testing engine.


