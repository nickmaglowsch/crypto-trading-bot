"""
crypto_market_neutral_backtest.py

Backtests the market-neutral momentum-driven crypto strategy you described:
- Universe: Top 10 market-cap coins (snapshot at each weekly rebalance)
- Momentum: risk-adjusted momentum = (lookback_return / lookback_vol)
- Selection: long top 3 momentum, short bottom 3 momentum (from top-10 universe)
- Sizing: inverse-volatility weighting across the selected 6 coins using the same 100k capital pool
- Rebalance: weekly
- No stop-loss

Notes:
- Uses Binance API via ccxt to fetch historical OHLCV data.
- Market cap is approximated using price * volume as a proxy.
- Top coins are selected by 24h trading volume (as proxy for market cap).
- No fees, no slippage, allows synthetic shorting (uses returns to apply negative exposure).

Requirements:
pip install pandas numpy ccxt tqdm python-dateutil

Run:
python src/test.py

"""

import time
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
import ccxt
import sys
from pathlib import Path

# Add parent directory to path to allow imports when running from src/
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    from src.binance_client import BinanceConfig, create_client
    from src.data_fetcher import fetch_ohlcv_dataframe
except ImportError:
    # Fallback for when running from src/ directory
    from binance_client import BinanceConfig, create_client
    from data_fetcher import fetch_ohlcv_dataframe

# ---------------------------
# Parameters
# ---------------------------
INITIAL_CAPITAL = 100000.0
START_DATE = "2016-01-01"    # YYYY-MM-DD
END_DATE = "2025-11-24"      # YYYY-MM-DD
CANDIDATE_TOP_N = 100         # fetch current top N to consider for historical universe (increased to account for stablecoin filtering)
QUOTE_CURRENCY = "USDT"       # quote currency for trading pairs
MOMENTUM_LOOKBACK_DAYS = 60   # for return (used in 'single' mode)
VOL_LOOKBACK_DAYS = 60        # for vol
REBALANCE_FREQ_DAYS = 7       # weekly
MIN_PRICE_POINTS = 30         # minimum historical points required
TIMEFRAME = "1d"              # Binance timeframe for daily data
MOMENTUM_METHOD = "multi"     # 'single' for risk-adjusted momentum, 'multi' for weekly momentum

# Stablecoins to exclude from trading (base assets only, not quote currency)
STABLECOINS = ['USDC', 'FDUSD', 'BUSD', 'DAI', 'TUSD', 'USDP', 'USDD', 'EUR', 'GBP', 'AUD', 'CAD', 'JPY', 'CHF', 'PAXG']
# Note: USDT is the quote currency, so we don't exclude it as a base asset

# ---------------------------
# Helper: Binance API helpers
# ---------------------------

def get_top_coins_by_volume(quote_currency="USDT", limit=50, exclude_assets=None):
    """
    Get top coins by 24h trading volume from Binance.
    Returns list of tuples: (symbol, base_asset, quote_asset)
    
    Parameters:
    -----------
    exclude_assets: list of asset symbols to exclude (e.g., ['FDUSD', 'EUR', 'USDC'])
    """
    if exclude_assets is None:
        exclude_assets = []
    
    client = create_client(BinanceConfig())
    tickers = client.fetch_tickers()
    
    # Filter for pairs with the quote currency and sort by 24h volume
    relevant_tickers = []
    for symbol, ticker in tickers.items():
        if symbol.endswith(f"/{quote_currency}") and ticker.get('quoteVolume', 0) > 0:
            base_asset = symbol.split('/')[0]
            # Exclude specified assets (case-insensitive)
            if base_asset.upper() in [a.upper() for a in exclude_assets]:
                continue
            relevant_tickers.append({
                'symbol': symbol,
                'base': base_asset,
                'quote': quote_currency,
                'volume24h': ticker.get('quoteVolume', 0)
            })
    
    # Sort by volume and return top N
    relevant_tickers.sort(key=lambda x: x['volume24h'], reverse=True)
    return relevant_tickers[:limit]


# ---------------------------
# Step 2: Fetch OHLCV data for a coin and compute market cap proxy
# Uses Binance OHLCV data via data_fetcher
# Returns DataFrame indexed by UTC date with columns: price, market_cap (proxy)
# ---------------------------

def fetch_coin_timeseries(symbol, from_dt: datetime, to_dt: datetime):
    """
    Fetch OHLCV data for a symbol and compute market cap proxy (price * volume).
    Returns DataFrame with columns: price, market_cap
    """
    try:
        # Ensure timezone-aware datetimes
        if from_dt.tzinfo is None:
            from_dt = from_dt.replace(tzinfo=timezone.utc)
        if to_dt.tzinfo is None:
            to_dt = to_dt.replace(tzinfo=timezone.utc)
        
        # Calculate days needed (add buffer for lookback)
        days = (to_dt - from_dt).days + VOL_LOOKBACK_DAYS + 10
        
        # Fetch OHLCV data
        ohlcv = fetch_ohlcv_dataframe(
            symbol=symbol,
            timeframe=TIMEFRAME,
            days=days,
            end_time=to_dt,
            config=BinanceConfig()
        )
        
        if ohlcv.empty:
            return pd.DataFrame()
        
        # Extract price (close) and compute market cap proxy (price * volume)
        df = pd.DataFrame(index=ohlcv.index)
        df['price'] = ohlcv['close']
        df['market_cap'] = ohlcv['close'] * ohlcv['volume']  # Proxy: price * volume
        
        # Filter to date range (ensure timezone-aware comparison)
        df = df[(df.index >= from_dt) & (df.index <= to_dt)]
        
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()


# ---------------------------
# Utility: daily resample to get daily close price and market cap (UTC)
# ---------------------------

def daily_from_timeseries(df: pd.DataFrame):
    # df indexed by datetime
    if df.empty:
        return df
    daily = df.resample('1D').last().ffill()
    # remove timezone info for simplicity
    daily.index = daily.index.normalize()
    return daily


# ---------------------------
# Momentum & volatility computation
# ---------------------------

def compute_risk_adjusted_momentum(price_series: pd.Series, lookback_days=60, vol_days=60):
    """
    Original risk-adjusted momentum: (lookback_return / lookback_vol)
    """
    # price_series indexed by date (daily)
    ret = price_series.pct_change(periods=lookback_days)
    # volatility as std of daily returns over vol_days
    vol = price_series.pct_change().rolling(vol_days).std()
    # Align by lookback (use end date's return and vol)
    mom = ret / vol
    return mom


def compute_multi_timeframe_momentum(price_series: pd.Series, vol_days=60):
    """
    Weekly momentum: 7-day return normalized by volatility.
    
    Returns risk-adjusted momentum score.
    """
    # Calculate 7-day (1 week) return
    ret_1w = price_series.pct_change(periods=7)
    
    # Calculate volatility (std of daily returns)
    vol = price_series.pct_change().rolling(vol_days).std()
    
    # Risk-adjusted momentum: weekly return / volatility
    mom = ret_1w / vol
    
    return mom


# ---------------------------
# Backtest core
# ---------------------------

def backtest(start_date, end_date, initial_capital=100000.0):
    start_dt = pd.to_datetime(start_date).tz_localize('UTC')
    end_dt = pd.to_datetime(end_date).tz_localize('UTC')

    # 1) candidates
    print("Fetching current top coins list (candidate pool) from Binance...")
    print(f"Excluding stablecoins: {STABLECOINS}")
    top_coins = get_top_coins_by_volume(QUOTE_CURRENCY, limit=CANDIDATE_TOP_N, exclude_assets=STABLECOINS)
    candidates = [(coin['symbol'], coin['base'], coin['base']) for coin in top_coins]
    print(f"Got {len(candidates)} candidate coins (after excluding stablecoins)")

    # 2) fetch timeseries for each candidate
    coin_data = {}
    print("Fetching historical price & market-cap time series for each candidate (this can take a while)")
    for symbol, base, name in tqdm(candidates):
        try:
            df = fetch_coin_timeseries(symbol, start_dt - timedelta(days=VOL_LOOKBACK_DAYS+10), end_dt + timedelta(days=1))
            if df.empty or len(df) < MIN_PRICE_POINTS:
                continue
            daily = daily_from_timeseries(df)
            coin_data[symbol] = {
                'symbol': base,
                'name': name,
                'daily': daily
            }
            # Small delay to avoid rate limits
            time.sleep(0.1)
        except Exception as e:
            print(f"Failed fetching {symbol}: {e}")

    if not coin_data:
        raise RuntimeError("No coin data fetched. Try increasing CANDIDATE_TOP_N or check dates.")

    # 3) build a combined date index
    all_dates = pd.date_range(start=start_dt.normalize(), end=end_dt.normalize(), freq='D', tz='UTC')

    # 4) For each coin, ensure we have a daily price series over the entire period (ffill)
    prices = pd.DataFrame(index=all_dates)
    mcaps = pd.DataFrame(index=all_dates)
    for symbol, info in coin_data.items():
        s_price = info['daily']['price'].reindex(all_dates).ffill()
        s_mcap = info['daily']['market_cap'].reindex(all_dates).ffill()
        prices[symbol] = s_price
        mcaps[symbol] = s_mcap

    prices = prices.dropna(axis=1, how='all')
    mcaps = mcaps[prices.columns]

    # 5) Backtest loop: weekly rebalances
    portfolio_value = initial_capital
    portfolio_values = []
    positions = {}  # coin_id -> position fraction (signed) at last rebalance

    # Determine minimum lookback days based on momentum method
    if MOMENTUM_METHOD == "multi":
        min_lookback_days = max(7, VOL_LOOKBACK_DAYS)  # Need 7 days for weekly momentum
    else:
        min_lookback_days = max(MOMENTUM_LOOKBACK_DAYS, VOL_LOOKBACK_DAYS)
    
    rebalance_dates = pd.date_range(start=start_dt.normalize(), end=end_dt.normalize(), freq=f"{REBALANCE_FREQ_DAYS}D", tz='UTC')
    # ensure we start after enough lookback for momentum calculation
    rebalance_dates = rebalance_dates[rebalance_dates >= (start_dt.normalize() + pd.Timedelta(days=min_lookback_days))]
    
    print(f"Using momentum method: {MOMENTUM_METHOD}")
    if MOMENTUM_METHOD == "multi":
        print("  Weekly momentum: 7-day risk-adjusted return")
    else:
        print(f"  Single timeframe: {MOMENTUM_LOOKBACK_DAYS}-day risk-adjusted momentum")

    current_positions = {c: 0.0 for c in prices.columns}
    position_types = {c: None for c in prices.columns}  # Track if position is 'long' or 'short'

    # We'll track daily PnL by applying positions to daily returns
    daily_returns = prices.pct_change().fillna(0)
    
    # Track detailed statistics
    daily_long_returns = []  # Daily returns from long positions
    daily_short_returns = []  # Daily returns from short positions
    daily_long_dates = []  # Dates for long returns
    daily_short_dates = []  # Dates for short returns
    asset_returns = {c: [] for c in prices.columns}  # Returns by asset
    asset_long_returns = {c: [] for c in prices.columns}  # Returns when asset was long
    asset_short_returns = {c: [] for c in prices.columns}  # Returns when asset was short
    asset_dates = {c: [] for c in prices.columns}  # Dates when asset had positions
    asset_long_dates = {c: [] for c in prices.columns}  # Dates when asset was long
    asset_short_dates = {c: [] for c in prices.columns}  # Dates when asset was short

    next_rebalance_idx = 0
    for today in all_dates:
        # Rebalance if today matches a rebalance date
        if next_rebalance_idx < len(rebalance_dates) and today == rebalance_dates[next_rebalance_idx]:
            # determine universe: compute market cap at this date and pick top 15 (exclude NaNs)
            # We get 15 so that after filtering stablecoins we still have at least 10
            mcap_today = mcaps.loc[today].dropna()
            top15 = mcap_today.sort_values(ascending=False).head(15).index.tolist()
            
            # Filter out stablecoins if they somehow made it through
            # Extract base asset from symbol (e.g., "BTC/USDT" -> "BTC")
            top15_filtered = []
            for coin in top15:
                # Extract base asset (part before the slash)
                base_asset = coin.split('/')[0] if '/' in coin else coin
                # Check if base asset is a stablecoin
                if base_asset.upper() not in [s.upper() for s in STABLECOINS]:
                    top15_filtered.append(coin)
            
            # Use top 10 from the filtered list
            top10 = top15_filtered[:10]
            
            if len(top10) < 10:
                print(f"Warning: Only {len(top10)} coins available after filtering on {today.date()}")
            if len(top10) == 0:
                print(f"ERROR: No coins available after filtering on {today.date()}. Available coins: {top15[:5]}")
                # Keep existing positions and skip this rebalance
                next_rebalance_idx += 1
                continue

            # compute risk-adjusted momentum for each top10 coin using lookback window ending today
            mom_scores = {}
            vols = {}
            for coin in top10:
                price_series = prices[coin].loc[:today]
                
                # Check minimum data requirements based on momentum method
                if MOMENTUM_METHOD == "multi":
                    min_days_required = max(7, VOL_LOOKBACK_DAYS)  # Need at least 7 days for weekly momentum
                else:
                    min_days_required = max(MOMENTUM_LOOKBACK_DAYS, VOL_LOOKBACK_DAYS)
                
                if len(price_series.dropna()) < min_days_required:
                    continue
                
                # use the series up to today
                if MOMENTUM_METHOD == "multi":
                    mom_series = compute_multi_timeframe_momentum(price_series, VOL_LOOKBACK_DAYS)
                else:
                    mom_series = compute_risk_adjusted_momentum(price_series, MOMENTUM_LOOKBACK_DAYS, VOL_LOOKBACK_DAYS)
                
                mom_val = mom_series.loc[today]
                # compute volatility (std of daily returns over vol window)
                vol_val = price_series.pct_change().rolling(VOL_LOOKBACK_DAYS).std().loc[today]
                mom_scores[coin] = mom_val
                vols[coin] = vol_val

            # drop coins with NaN scores
            mom_scores = {k: v for k, v in mom_scores.items() if pd.notna(v) and pd.notna(vols.get(k)) and vols.get(k) > 0}
            if len(mom_scores) < 3:
                print(f"Insufficient momentum data on {today.date()}, skipping rebalance")
                # keep existing positions
            else:
                # rank by mom
                sorted_coins = sorted(mom_scores.items(), key=lambda x: x[1], reverse=True)
                longs = [c for c, s in sorted_coins[:3]]
                shorts = [c for c, s in sorted_coins[-3:]]
                selected = longs + shorts

                # build inverse-vol weights separately for longs and shorts to ensure market neutrality
                # Longs: normalize to sum to 0.5 (50% of capital)
                iv_longs = {c: 1.0 / vols[c] for c in longs}
                total_iv_longs = sum(iv_longs.values())
                weights_longs = {c: (iv_longs[c] / total_iv_longs) * 0.5 if total_iv_longs > 0 else 0.0 for c in longs}
                
                # Shorts: normalize to sum to -0.5 (50% of capital, negative)
                iv_shorts = {c: 1.0 / vols[c] for c in shorts}
                total_iv_shorts = sum(iv_shorts.values())
                weights_shorts = {c: (iv_shorts[c] / total_iv_shorts) * -0.5 if total_iv_shorts > 0 else 0.0 for c in shorts}
                
                # combine signed weights (longs positive, shorts negative)
                signed_weights = {**weights_longs, **weights_shorts}

                # convert to position fractions relative to current portfolio value
                # We'll store position fraction as dollar exposure / portfolio_value
                current_positions = {coin: 0.0 for coin in prices.columns}
                position_types = {coin: None for coin in prices.columns}
                for c, w in signed_weights.items():
                    current_positions[c] = w
                    position_types[c] = 'long' if w > 0 else 'short'

                print(f"{today.date()}: Rebalanced. Longs: {longs}, Shorts: {shorts}")

            next_rebalance_idx += 1

        # apply daily returns to update portfolio value
        # portfolio_return = sum(position_fraction * daily_return)  (since position fractions sum to net exposure)
        today_ret = 0.0
        today_long_ret = 0.0
        today_short_ret = 0.0
        
        for coin, pos_frac in current_positions.items():
            if pos_frac == 0:
                continue
            coin_ret = daily_returns.at[today, coin]
            if pd.isna(coin_ret):
                coin_ret = 0.0
            
            # Contribution to total return
            contribution = pos_frac * coin_ret
            today_ret += contribution
            
            # Track by position type
            if pos_frac > 0:  # Long position
                today_long_ret += contribution
                asset_long_returns[coin].append(coin_ret)
                asset_long_dates[coin].append(today)
            else:  # Short position
                today_short_ret += contribution
                asset_short_returns[coin].append(coin_ret)
                asset_short_dates[coin].append(today)
            
            # Track by asset
            asset_returns[coin].append(coin_ret)
            asset_dates[coin].append(today)
        
        daily_long_returns.append(today_long_ret)
        daily_short_returns.append(today_short_ret)
        daily_long_dates.append(today)
        daily_short_dates.append(today)

        # update portfolio value: simple linear approximation
        portfolio_value = portfolio_value * (1.0 + today_ret)
        portfolio_values.append({'date': today, 'portfolio_value': portfolio_value})

    results = pd.DataFrame(portfolio_values).set_index('date')
    
    # Create detailed statistics dictionary
    stats = {
        'daily_long_returns': pd.Series(daily_long_returns, index=daily_long_dates) if daily_long_returns else pd.Series(dtype=float),
        'daily_short_returns': pd.Series(daily_short_returns, index=daily_short_dates) if daily_short_returns else pd.Series(dtype=float),
        'asset_returns': {k: pd.Series(v, index=asset_dates[k]) for k, v in asset_returns.items() if v},
        'asset_long_returns': {k: pd.Series(v, index=asset_long_dates[k]) for k, v in asset_long_returns.items() if v and len(asset_long_dates[k]) == len(v)},
        'asset_short_returns': {k: pd.Series(v, index=asset_short_dates[k]) for k, v in asset_short_returns.items() if v and len(asset_short_dates[k]) == len(v)},
    }
    
    return results, prices, mcaps, stats


# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    print("Starting backtest...")
    results, prices, mcaps, stats = backtest(START_DATE, END_DATE, INITIAL_CAPITAL)
    print("Backtest complete.\n")

    # simple performance metrics
    results = results.dropna()
    results['daily_ret'] = results['portfolio_value'].pct_change().fillna(0)
    total_return = results['portfolio_value'].iloc[-1] / results['portfolio_value'].iloc[0] - 1
    ann_return = (1 + total_return) ** (365.0 / len(results)) - 1
    ann_vol = results['daily_ret'].std() * math.sqrt(365)
    sharpe = (ann_return / ann_vol) if ann_vol > 0 else np.nan

    print("=" * 80)
    print("OVERALL PERFORMANCE")
    print("=" * 80)
    print(f"Initial capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Final capital: ${results['portfolio_value'].iloc[-1]:,.2f}")
    print(f"Total return: {total_return:.2%}")
    print(f"Total return (multiple): {results['portfolio_value'].iloc[-1] / INITIAL_CAPITAL:.2f}x")
    print(f"Annualized return: {ann_return:.2%}")
    print(f"Annualized vol: {ann_vol:.2%}")
    print(f"Sharpe (rf=0): {sharpe:.2f}")
    
    # Calculate maximum drawdown
    portfolio_values = results['portfolio_value']
    running_max = portfolio_values.cummax()
    drawdown = (portfolio_values - running_max) / running_max
    max_drawdown = drawdown.min()
    max_drawdown_date = drawdown.idxmin()
    
    # Find the peak before the max drawdown
    peak_value = running_max.loc[max_drawdown_date]
    trough_value = portfolio_values.loc[max_drawdown_date]
    
    # Find when the peak was first reached
    peak_date = portfolio_values[portfolio_values == peak_value].index[0]
    
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"  Peak: ${peak_value:,.2f} on {peak_date.date()}")
    print(f"  Trough: ${trough_value:,.2f} on {max_drawdown_date.date()}")
    print(f"  Recovery: ${portfolio_values.iloc[-1]:,.2f} (current)")
    
    # Long vs Short performance
    print("\n" + "=" * 80)
    print("LONG vs SHORT PERFORMANCE")
    print("=" * 80)
    
    long_returns = stats['daily_long_returns']
    short_returns = stats['daily_short_returns']
    
    if len(long_returns) > 0:
        long_total_ret = (1 + long_returns).prod() - 1
        long_ann_ret = (1 + long_total_ret) ** (365.0 / len(long_returns)) - 1
        long_ann_vol = long_returns.std() * math.sqrt(365)
        long_sharpe = (long_ann_ret / long_ann_vol) if long_ann_vol > 0 else np.nan
        long_win_rate = (long_returns > 0).sum() / len(long_returns) if len(long_returns) > 0 else 0
        
        print(f"\nLONG POSITIONS:")
        print(f"  Total return contribution: {long_total_ret:.2%}")
        print(f"  Annualized return: {long_ann_ret:.2%}")
        print(f"  Annualized volatility: {long_ann_vol:.2%}")
        print(f"  Sharpe ratio: {long_sharpe:.2f}")
        print(f"  Win rate (positive days): {long_win_rate:.2%}")
        print(f"  Average daily return: {long_returns.mean():.4%}")
        print(f"  Best day: {long_returns.max():.2%}")
        print(f"  Worst day: {long_returns.min():.2%}")
    
    if len(short_returns) > 0:
        short_total_ret = (1 + short_returns).prod() - 1
        short_ann_ret = (1 + short_total_ret) ** (365.0 / len(short_returns)) - 1
        short_ann_vol = short_returns.std() * math.sqrt(365)
        short_sharpe = (short_ann_ret / short_ann_vol) if short_ann_vol > 0 else np.nan
        short_win_rate = (short_returns > 0).sum() / len(short_returns) if len(short_returns) > 0 else 0
        
        print(f"\nSHORT POSITIONS:")
        print(f"  Total return contribution: {short_total_ret:.2%}")
        print(f"  Annualized return: {short_ann_ret:.2%}")
        print(f"  Annualized volatility: {short_ann_vol:.2%}")
        print(f"  Sharpe ratio: {short_sharpe:.2f}")
        print(f"  Win rate (positive days): {short_win_rate:.2%}")
        print(f"  Average daily return: {short_returns.mean():.4%}")
        print(f"  Best day: {short_returns.max():.2%}")
        print(f"  Worst day: {short_returns.min():.2%}")
    
    # Performance by asset
    print("\n" + "=" * 80)
    print("PERFORMANCE BY ASSET")
    print("=" * 80)
    
    asset_performance = []
    for asset, returns in stats['asset_returns'].items():
        if len(returns) == 0:
            continue
        
        asset_name = asset.split('/')[0] if '/' in asset else asset
        total_ret = (1 + returns).prod() - 1
        ann_ret = (1 + total_ret) ** (365.0 / len(returns)) - 1 if len(returns) > 0 else 0
        ann_vol = returns.std() * math.sqrt(365) if len(returns) > 0 else 0
        sharpe_asset = (ann_ret / ann_vol) if ann_vol > 0 else np.nan
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        # Long performance for this asset
        long_ret_asset = stats['asset_long_returns'].get(asset, pd.Series(dtype=float))
        short_ret_asset = stats['asset_short_returns'].get(asset, pd.Series(dtype=float))
        
        long_total_ret_asset = (1 + long_ret_asset).prod() - 1 if len(long_ret_asset) > 0 else 0
        short_total_ret_asset = (1 + short_ret_asset).prod() - 1 if len(short_ret_asset) > 0 else 0
        
        asset_performance.append({
            'asset': asset_name,
            'total_return': total_ret,
            'ann_return': ann_ret,
            'ann_vol': ann_vol,
            'sharpe': sharpe_asset,
            'win_rate': win_rate,
            'days': len(returns),
            'long_return': long_total_ret_asset,
            'short_return': short_total_ret_asset,
            'long_days': len(long_ret_asset),
            'short_days': len(short_ret_asset),
        })
    
    # Sort by total return
    asset_performance.sort(key=lambda x: x['total_return'], reverse=True)
    
    print(f"\n{'Asset':<15} {'Total Ret':<12} {'Ann Ret':<12} {'Sharpe':<8} {'Win Rate':<10} {'Days':<8} {'Long Ret':<12} {'Short Ret':<12}")
    print("-" * 100)
    for perf in asset_performance[:20]:  # Top 20 assets
        print(f"{perf['asset']:<15} {perf['total_return']:>11.2%} {perf['ann_return']:>11.2%} "
              f"{perf['sharpe']:>7.2f} {perf['win_rate']:>9.2%} {perf['days']:>7} "
              f"{perf['long_return']:>11.2%} {perf['short_return']:>11.2%}")
    
    # Detailed asset breakdown
    print("\n" + "=" * 80)
    print("DETAILED ASSET BREAKDOWN (Top 10)")
    print("=" * 80)
    
    for perf in asset_performance[:10]:
        asset_name = perf['asset']
        asset_key = next((k for k in stats['asset_returns'].keys() if k.split('/')[0] == asset_name), None)
        if not asset_key:
            continue
        
        print(f"\n{asset_name}:")
        print(f"  Overall: {perf['total_return']:.2%} total return, {perf['ann_return']:.2%} annualized, "
              f"{perf['sharpe']:.2f} Sharpe, {perf['win_rate']:.2%} win rate over {perf['days']} days")
        
        if perf['long_days'] > 0 and asset_key in stats['asset_long_returns']:
            long_ret_asset = stats['asset_long_returns'][asset_key]
            long_ann_ret = (1 + perf['long_return']) ** (365.0 / perf['long_days']) - 1 if perf['long_days'] > 0 else 0
            long_win_rate = (long_ret_asset > 0).sum() / len(long_ret_asset) if len(long_ret_asset) > 0 else 0
            print(f"  As LONG: {perf['long_return']:.2%} total return, {long_ann_ret:.2%} annualized, "
                  f"{long_win_rate:.2%} win rate over {perf['long_days']} days")
        
        if perf['short_days'] > 0 and asset_key in stats['asset_short_returns']:
            short_ret_asset = stats['asset_short_returns'][asset_key]
            short_ann_ret = (1 + perf['short_return']) ** (365.0 / perf['short_days']) - 1 if perf['short_days'] > 0 else 0
            short_win_rate = (short_ret_asset > 0).sum() / len(short_ret_asset) if len(short_ret_asset) > 0 else 0
            print(f"  As SHORT: {perf['short_return']:.2%} total return, {short_ann_ret:.2%} annualized, "
                  f"{short_win_rate:.2%} win rate over {perf['short_days']} days")

    # Save results
    results.to_csv('backtest_results.csv')
    print(f"\nSaved backtest_results.csv")
