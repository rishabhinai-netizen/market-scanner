import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="ProTrader - Twin Engine Pro", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# CSS: Professional UI
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        div.stButton > button { 
            background-color: #00C853; color: white; font-weight: bold; border: none; width: 100%; 
        }
        .metric-box { 
            padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 10px; border: 1px solid #ddd; 
        }
        .buy-signal { background-color: #d4edda; color: #155724; border-color: #c3e6cb; }
        .sell-signal { background-color: #f8d7da; color: #721c24; border-color: #f5c6cb; }
        .neutral { background-color: #f8f9fa; color: #666; }
        .bull-regime { background-color: #d4edda; border-left: 4px solid #28a745; }
        .bear-regime { background-color: #f8d7da; border-left: 4px solid #dc3545; }
        .choppy-regime { background-color: #fff3cd; border-left: 4px solid #ffc107; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ ProTrader: Twin-Engine Pro (v2.0)")

# --- 2. MARKET REGIME DEFINITIONS ---
MARKET_PERIODS = {
    "COVID Crash (2020)": {"start": "2020-01-01", "end": "2020-04-30", "type": "BEAR", "description": "Extreme volatility, -40% drawdown"},
    "V-Recovery (2020)": {"start": "2020-05-01", "end": "2020-12-31", "type": "BULL", "description": "Sharp recovery rally"},
    "Post-COVID Bull (2021)": {"start": "2021-01-01", "end": "2021-12-31", "type": "BULL", "description": "Liquidity-driven bull run"},
    "Rate Hike Chop (2022)": {"start": "2022-01-01", "end": "2022-12-31", "type": "CHOPPY", "description": "Sideways grind, FII selling"},
    "Steady Bull (2023)": {"start": "2023-01-01", "end": "2023-12-31", "type": "BULL", "description": "DII-driven rally"},
    "Current (2024+)": {"start": "2024-01-01", "end": datetime.now().strftime("%Y-%m-%d"), "type": "BULL", "description": "Ongoing market"},
    "Full Dataset (2Y)": {"start": (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d"), 
                          "end": datetime.now().strftime("%Y-%m-%d"), "type": "MIXED", "description": "Last 2 years"}
}

# --- 3. ASSET LISTS ---
COMMODITIES = {
    'Gold (Global)': 'GC=F', 'Silver (Global)': 'SI=F', 'Copper (Global)': 'HG=F',
    'Crude Oil': 'CL=F', 'Natural Gas': 'NG=F'
}

# Reduced NSE list for demo (use full list in production)
NSE_500_LIST = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS',
    'BHARTIARTL.NS', 'KOTAKBANK.NS', 'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS',
    'SUNPHARMA.NS', 'BAJFINANCE.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TATASTEEL.NS',
    'POWERGRID.NS', 'NTPC.NS', 'COALINDIA.NS', 'ONGC.NS', 'M&M.NS', 'ADANIPORTS.NS', 'TATAMOTORS.NS', 'JSWSTEEL.NS',
    # Add more stocks here - truncated for brevity
]

# --- 4. CORE ANALYTICS ENGINE ---

@st.cache_data(ttl=3600)
def fetch_nifty_data(start_date=None, end_date=None):
    """Downloads Nifty 50 for Relative Strength + Regime Detection"""
    try:
        if start_date and end_date:
            nifty = yf.download("^NSEI", start=start_date, end=end_date, progress=False, auto_adjust=True)
        else:
            nifty = yf.download("^NSEI", period="3y", progress=False, auto_adjust=True)
        
        # Calculate ROC for RS
        nifty['ROC_55'] = nifty['Close'].pct_change(periods=55) * 100
        
        # Calculate regime indicators
        nifty['SMA_50'] = nifty['Close'].rolling(50).mean()
        nifty['SMA_200'] = nifty['Close'].rolling(200).mean()
        
        return nifty
    except:
        return None

def calculate_regime_score(df, nifty_df=None):
    """
    Adaptive Regime Detection - Returns score 0-100
    >60: Strong trend (favorable)
    40-60: Neutral/Choppy
    <40: Weak/Bear (unfavorable)
    """
    
    # Factor 1: Trend Strength (40%)
    sma_20 = df['Close'].rolling(20).mean()
    sma_50 = df['Close'].rolling(50).mean()
    sma_200 = df['Close'].rolling(200).mean()
    
    alignment = (
        ((sma_20 > sma_50).astype(int) * 50) +
        ((sma_50 > sma_200).astype(int) * 50)
    )
    
    price_vs_sma200 = ((df['Close'] - sma_200) / sma_200 * 100).clip(-20, 20)
    price_score = (price_vs_sma200 + 20) / 40 * 100
    
    trend_strength = (alignment * 0.6 + price_score * 0.4)
    
    # Factor 2: Volatility Regime (30%)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    atr_pct = (atr_14 / df['Close']) * 100
    
    atr_percentile = atr_pct.rolling(100).apply(
        lambda x: (x[-1] <= x).sum() / len(x) * 100, raw=True
    )
    
    volatility_score = 100 - atr_percentile
    
    # Factor 3: Consistency (30%)
    time_index = pd.Series(range(len(df)), index=df.index)
    rolling_corr = df['Close'].rolling(20).corr(time_index)
    consistency_score = (rolling_corr.abs() * 100).fillna(50)
    
    # Composite Score
    regime_score = (
        trend_strength * 0.40 +
        volatility_score * 0.30 +
        consistency_score * 0.30
    ).fillna(50)
    
    return regime_score.rolling(5).mean()

def calculate_indicators(df, nifty_roc=None, regime_enabled=False, nifty_df=None):
    """Enhanced indicator calculation with regime detection"""
    
    # 1. Donchian
    df['High_20'] = df['High'].rolling(20).max()
    df['Low_20'] = df['Low'].rolling(20).min()
    df['Middle'] = (df['High_20'] + df['Low_20']) / 2
    
    # 2. Trend EMAs
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # 3. RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Vol_SMA_7'] = df['Volume'].rolling(7).mean().shift(1)
    
    # 4. ATR for Adaptive Stops
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()
    
    # 5. Relative Strength (RS) - Fixed timing
    if nifty_roc is not None:
        df['Stock_ROC_55'] = df['Close'].pct_change(periods=55) * 100
        df['Nifty_ROC'] = nifty_roc.reindex(df.index).ffill()
        # LAG by 1 day to avoid look-ahead bias
        df['RS_Pass'] = (df['Stock_ROC_55'] > df['Nifty_ROC']).shift(1).fillna(False)
    else:
        df['RS_Pass'] = True
    
    # 6. Regime Score
    if regime_enabled:
        df['Regime_Score'] = calculate_regime_score(df, nifty_df)
        df['Regime_OK'] = df['Regime_Score'] > 55
    else:
        df['Regime_OK'] = True
    
    return df

# --- 5. VECTORIZED BACKTEST ENGINES ---

def get_dc_stats_vectorized(df, params):
    """STRATEGY 1: Donchian Breakout (Vectorized)"""
    
    prev_close = df['Close'].shift(1)
    prev_mid = df['Middle'].shift(1)
    
    # Base Signals
    cross_up = (df['Close'] > df['Middle']) & (prev_close < prev_mid)
    cross_down = (df['Close'] < df['Middle']) & (prev_close > prev_mid)
    
    entry_signal = cross_up.copy()
    
    # Apply Filters
    if params['use_vol']:
        vol_cond = (df['Volume'] > params['vol_mult'] * df['Vol_SMA_7']) & (df['Vol_SMA_7'] > params['min_liq'])
        entry_signal = entry_signal & vol_cond
        
    if params['use_rs']:
        entry_signal = entry_signal & df['RS_Pass']
    
    if params.get('regime_enabled', False):
        entry_signal = entry_signal & df['Regime_OK']
    
    # Position Logic
    position_changes = pd.Series(0, index=df.index)
    position_changes[entry_signal] = 1
    position_changes[cross_down] = -1
    
    raw_position = position_changes.cumsum()
    raw_position = raw_position - raw_position.iloc[0]
    position = raw_position.clip(0, 1).shift(1).fillna(0)
    
    # Calculate Returns
    df['Strat_Ret'] = df['Close'].pct_change() * position
    
    # Apply Transaction Costs (0.25% per trade)
    signal_changes = position.diff().abs() > 0
    transaction_cost = params.get('transaction_cost', 0.0025)
    df.loc[signal_changes, 'Strat_Ret'] -= transaction_cost
    
    cum_ret = (1 + df['Strat_Ret']).cumprod().iloc[-1] - 1
    trades = position.diff().abs().sum() / 2
    
    return round(cum_ret * 100, 2), int(trades)

def get_trend_surfer_stats_vectorized(df, params):
    """STRATEGY 2: Trend Surfer (Fully Vectorized + Adaptive Stops)"""
    
    use_adaptive_stop = params.get('use_adaptive_stop', False)
    stop_loss_pct = params['stop_loss']
    use_rs = params['use_rs']
    regime_enabled = params.get('regime_enabled', False)
    
    valid_start = 55
    if len(df) < valid_start:
        return 0.0, 0
    
    df_bt = df.iloc[valid_start:].copy()
    
    # Entry Conditions
    trend_condition = (df_bt['Close'] > df_bt['EMA_20']) & (df_bt['Close'] > df_bt['EMA_50'])
    
    if use_rs:
        trend_condition = trend_condition & df_bt['RS_Pass']
    
    if regime_enabled:
        trend_condition = trend_condition & df_bt['Regime_OK']
    
    entry_signal = trend_condition
    
    # Exit Condition (Trend Break)
    exit_signal = df_bt['Close'] < df_bt['EMA_20']
    
    # Position State Machine
    position_changes = pd.Series(0, index=df_bt.index)
    position_changes[entry_signal] = 1
    position_changes[exit_signal] = -1
    
    raw_position = position_changes.cumsum()
    raw_position = raw_position - raw_position.iloc[0]
    position_state = raw_position.clip(0, 1)
    
    # Calculate Entry Prices
    entry_transitions = (position_state == 1) & (position_state.shift(1).fillna(0) == 0)
    entry_price = df_bt['Close'].where(entry_transitions).ffill()
    
    # Adaptive or Fixed Stop-Loss
    if use_adaptive_stop:
        # Stop = Entry - (2.5 × ATR)
        atr_multiplier = params.get('atr_multiplier', 2.5)
        stop_distance_pct = (df_bt['ATR_14'] / df_bt['Close']) * atr_multiplier * 100
        stop_price = entry_price * (1 - stop_distance_pct / 100)
    else:
        stop_price = entry_price * (1 - stop_loss_pct / 100)
    
    # Stop Hit Detection
    stop_hit = (position_state == 1) & (df_bt['Low'] < stop_price)
    
    # Final Exit (Stop OR Trend Break)
    final_exit = exit_signal | stop_hit
    
    # Rebuild Position with Stops
    position_changes_v2 = pd.Series(0, index=df_bt.index)
    position_changes_v2[entry_signal] = 1
    position_changes_v2[final_exit] = -1
    
    raw_position_v2 = position_changes_v2.cumsum()
    raw_position_v2 = raw_position_v2 - raw_position_v2.iloc[0]
    position_state_v2 = raw_position_v2.clip(0, 1)
    
    # Recalculate Entry Prices
    entry_transitions_v2 = (position_state_v2 == 1) & (position_state_v2.shift(1).fillna(0) == 0)
    entry_price_v2 = df_bt['Close'].where(entry_transitions_v2).ffill()
    
    # Calculate Exit Prices
    exit_transitions = (position_state_v2 == 0) & (position_state_v2.shift(1).fillna(0) == 1)
    
    if use_adaptive_stop:
        stop_price_v2 = entry_price_v2 * (1 - stop_distance_pct / 100)
    else:
        stop_price_v2 = entry_price_v2 * (1 - stop_loss_pct / 100)
    
    exit_price = pd.Series(index=df_bt.index, dtype=float)
    exit_price[exit_transitions & stop_hit] = stop_price_v2[exit_transitions & stop_hit]
    exit_price[exit_transitions & ~stop_hit] = df_bt['Close'][exit_transitions & ~stop_hit]
    
    # Calculate Returns
    returns_per_trade = ((exit_price - entry_price_v2) / entry_price_v2).dropna()
    
    # Apply Transaction Costs
    transaction_cost = params.get('transaction_cost', 0.0025)
    returns_per_trade = returns_per_trade - (transaction_cost * 2)  # Round-trip
    
    trades_count = len(returns_per_trade)
    
    if trades_count == 0:
        return 0.0, 0
    
    cumulative_return = (1 + returns_per_trade).prod() - 1
    total_return_pct = cumulative_return * 100
    
    return round(total_return_pct, 2), int(trades_count)

# --- 6. BULK PROCESSING ---

def process_batch(tickers, start_date=None, end_date=None):
    """Enhanced batch download with validation"""
    try:
        if start_date and end_date:
            data = yf.download(tickers, start=start_date, end=end_date, 
                             group_by='ticker', threads=True, progress=False, auto_adjust=True)
        else:
            data = yf.download(tickers, period="2y", 
                             group_by='ticker', threads=True, progress=False, auto_adjust=True)
        
        # Validate downloads
        if len(tickers) > 1 and not data.empty:
            downloaded = [t for t in tickers if t in data.columns.levels[0]]
            failed = set(tickers) - set(downloaded)
            if failed:
                st.warning(f"⚠️ Failed to download {len(failed)} stocks: {list(failed)[:5]}...")
        
        return data
    except Exception as e:
        st.error(f"Download error: {str(e)}")
        return None

# --- 7. REGIME INTELLIGENCE ENGINE ---

def detect_current_regime(nifty_df):
    """
    Analyzes current Nifty 50 data to classify market regime.
    Returns: (regime_type, confidence, description)
    """
    if nifty_df is None or len(nifty_df) < 200:
        return "UNKNOWN", 0, "Insufficient data"
    
    latest = nifty_df.iloc[-1]
    
    # Criteria 1: Price vs Moving Averages
    above_200sma = latest['Close'] > latest['SMA_200']
    above_50sma = latest['Close'] > latest['SMA_50']
    
    # Criteria 2: Trend Slope
    sma50_slope = (nifty_df['SMA_50'].iloc[-1] - nifty_df['SMA_50'].iloc[-20]) / nifty_df['SMA_50'].iloc[-20] * 100
    
    # Criteria 3: Volatility
    recent_returns = nifty_df['Close'].pct_change().iloc[-20:]
    volatility = recent_returns.std() * np.sqrt(252) * 100  # Annualized
    
    # Criteria 4: Recent Performance
    recent_return = (nifty_df['Close'].iloc[-1] / nifty_df['Close'].iloc[-60] - 1) * 100  # 3-month
    
    # Decision Logic
    confidence = 0
    
    if above_200sma and above_50sma and sma50_slope > 1 and recent_return > 5:
        regime_type = "BULL"
        confidence = 85
        description = f"Strong uptrend (+{recent_return:.1f}% in 3M). Vol: {volatility:.1f}%"
        
    elif above_200sma and above_50sma and sma50_slope > 0:
        regime_type = "BULL"
        confidence = 65
        description = f"Mild uptrend (+{recent_return:.1f}% in 3M). Vol: {volatility:.1f}%"
        
    elif not above_200sma and sma50_slope < -1 and recent_return < -5:
        regime_type = "BEAR"
        confidence = 80
        description = f"Downtrend ({recent_return:.1f}% in 3M). Vol: {volatility:.1f}%"
        
    elif volatility > 25 or abs(recent_return) < 3:
        regime_type = "CHOPPY"
        confidence = 70
        description = f"Sideways/Range-bound. High vol: {volatility:.1f}%"
        
    else:
        regime_type = "CHOPPY"
        confidence = 60
        description = f"Mixed signals. Return: {recent_return:.1f}%, Vol: {volatility:.1f}%"
    
    return regime_type, confidence, description

def recommend_strategy(regime_type, confidence):
    """
    Strategy recommendation based on current regime.
    """
    recommendations = {
        "BULL": {
            "primary": "Trend Surfer",
            "reason": "Strong trends favor EMA-based systems. Ride the momentum!",
            "settings": "Use Regime Filter ON, Adaptive Stops, RS Filter ON"
        },
        "BEAR": {
            "primary": "CASH / Defensive",
            "reason": "Both strategies are LONG-ONLY. Avoid trading in bear markets.",
            "settings": "Wait for regime change or consider inverse strategies"
        },
        "CHOPPY": {
            "primary": "Donchian Breakout",
            "reason": "Breakout systems can capture explosive moves in range-bound markets.",
            "settings": "Use strict Volume Filter (3x), disable Regime Filter"
        },
        "UNKNOWN": {
            "primary": "Wait",
            "reason": "Insufficient data to assess regime.",
            "settings": "Gather more data"
        }
    }
    
    rec = recommendations.get(regime_type, recommendations["UNKNOWN"])
    
    if confidence < 60:
        rec["reason"] += " ⚠️ LOW CONFIDENCE - Use caution!"
    
    return rec

# --- 8. MASTER SCANNER ---

@st.cache_data(ttl=900)
def run_master_scan(strategy_type, target_list, params, nifty_roc, start_date=None, end_date=None):
    results_buy = []
    results_sell = []
    
    chunk_size = 30
    total_chunks = list(range(0, len(target_list), chunk_size))
    progress = st.progress(0)
    status = st.empty()
    
    for i, start_idx in enumerate(total_chunks):
        batch = target_list[start_idx : start_idx + chunk_size]
        status.caption(f"🔍 Scanning batch {i+1}/{len(total_chunks)}...")
        
        data = process_batch(batch, start_date, end_date)
        if data is None or data.empty:
            continue
        
        for ticker in batch:
            try:
                df = data[ticker] if len(batch) > 1 else data.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.dropna()
                if len(df) < 200:
                    continue
                
                df = calculate_indicators(df, nifty_roc, params.get('regime_enabled', False))
                
                today = df.iloc[-1]
                prev = df.iloc[-2]
                display_name = ticker.replace('.NS', '').replace('=F', '')
                
                rs_valid = today['RS_Pass'] if params['use_rs'] else True
                regime_valid = today['Regime_OK'] if params.get('regime_enabled', False) else True
                
                # Strategy Branching
                if strategy_type == "Donchian Breakout":
                    is_cross_up = (prev['Close'] < prev['Middle']) and (today['Close'] > today['Middle'])
                    
                    if is_cross_up and rs_valid and regime_valid:
                        if params.get('trend') and today['Close'] < today['SMA_200']:
                            pass
                        elif params.get('rsi') and today['RSI'] > 70:
                            pass
                        else:
                            vol_spike = today['Volume'] / today['Vol_SMA_7'] if today['Vol_SMA_7'] > 0 else 0
                            valid_vol = True
                            if params['use_vol']:
                                if today['Vol_SMA_7'] < params['min_liq']:
                                    valid_vol = False
                                if vol_spike < params['vol_mult']:
                                    valid_vol = False
                            
                            if valid_vol:
                                ret, trades = get_dc_stats_vectorized(df, params)
                                results_buy.append({
                                    "Symbol": display_name,
                                    "Price": round(today['Close'], 2),
                                    "Signal": "Breakout",
                                    "Vol": f"{vol_spike:.1f}x",
                                    "Regime": f"{int(today['Regime_Score'])}" if 'Regime_Score' in today.index else "N/A",
                                    "Trades": trades,
                                    "Ret%": f"{ret}%",
                                    "raw_ret": ret
                                })
                    
                    is_cross_down = (prev['Close'] > prev['Middle']) and (today['Close'] < today['Middle'])
                    if is_cross_down:
                        results_sell.append({
                            "Symbol": display_name,
                            "Price": round(today['Close'], 2),
                            "Signal": "Exit (Below Mid)"
                        })
                
                elif strategy_type == "Trend Surfer":
                    is_uptrend = (today['Close'] > today['EMA_20']) and (today['Close'] > today['EMA_50'])
                    
                    if is_uptrend and rs_valid and regime_valid:
                        ret, trades = get_trend_surfer_stats_vectorized(df, params)
                        results_buy.append({
                            "Symbol": display_name,
                            "Price": round(today['Close'], 2),
                            "Signal": "Trend Strong",
                            "RS": "✓" if today['RS_Pass'] else "✗",
                            "Regime": f"{int(today['Regime_Score'])}" if 'Regime_Score' in today.index else "N/A",
                            "Trades": trades,
                            "Ret%": f"{ret}%",
                            "raw_ret": ret
                        })
                    
                    trend_broken = today['Close'] < today['EMA_20']
                    was_uptrend = prev['Close'] > prev['EMA_20']
                    if trend_broken and was_uptrend:
                        results_sell.append({
                            "Symbol": display_name,
                            "Price": round(today['Close'], 2),
                            "Signal": "Exit (Trend Break)"
                        })
            except:
                continue
        
        progress.progress((i + 1) / len(total_chunks))
    
    if results_buy:
        results_buy.sort(key=lambda x: x['raw_ret'], reverse=True)
        for item in results_buy:
            del item['raw_ret']
    
    progress.empty()
    status.empty()
    return results_buy, results_sell

# --- 9. BULK BACKTEST ---

@st.cache_data(ttl=3600)
def run_bulk_test(strategy_type, target_list, params, nifty_roc, start_date=None, end_date=None):
    prof, loss = [], []
    chunk_size = 50
    chunks = list(range(0, len(target_list), chunk_size))
    progress = st.progress(0)
    
    for i, start in enumerate(chunks):
        batch = target_list[start : start + chunk_size]
        data = process_batch(batch, start_date, end_date)
        if data is None or data.empty:
            continue
        
        for ticker in batch:
            try:
                df = data[ticker] if len(batch) > 1 else data.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.dropna()
                if len(df) < 200:
                    continue
                
                df = calculate_indicators(df, nifty_roc, params.get('regime_enabled', False))
                
                if strategy_type == "Donchian Breakout":
                    ret, trades = get_dc_stats_vectorized(df, params)
                else:
                    ret, trades = get_trend_surfer_stats_vectorized(df, params)
                
                entry = {
                    "Symbol": ticker.replace('.NS', ''),
                    "Return": f"{ret}%",
                    "Trades": trades,
                    "Raw_Ret": ret
                }
                
                if ret > 0:
                    prof.append(entry)
                else:
                    loss.append(entry)
            except:
                continue
        
        progress.progress((i + 1) / len(chunks))
    
    progress.empty()
    prof.sort(key=lambda x: x['Raw_Ret'], reverse=True)
    loss.sort(key=lambda x: x['Raw_Ret'])
    
    return prof, loss

# --- 10. UI LAYOUT ---

# SIDEBAR
with st.sidebar:
    st.header("🎮 Strategy Control")
    strat_mode = st.radio("Select Engine", ["Trend Surfer", "Donchian Breakout"])
    
    st.divider()
    
    # Global Filters
    st.subheader("🌍 Global Filters")
    use_rs = st.checkbox("✅ RS Filter (Stock > Nifty)", value=True)
    
    regime_enabled = st.checkbox(
        "🌊 Regime Filter (BETA)", 
        value=False,
        help="Only trade when market conditions favor trend-following"
    )
    
    if regime_enabled:
        regime_threshold = st.slider("Regime Threshold", 40, 70, 55)
    
    use_transaction_costs = st.checkbox("💰 Transaction Costs (0.25%)", value=True)
    
    st.divider()
    
    # Strategy-specific settings
    params = {
        'use_rs': use_rs,
        'regime_enabled': regime_enabled,
        'transaction_cost': 0.0025 if use_transaction_costs else 0.0
    }
    
    if strat_mode == "Donchian Breakout":
        st.subheader("🌊 Breakout Settings")
        params['trend'] = st.checkbox("Trend Filter (>200 SMA)", True)
        params['rsi'] = st.checkbox("RSI Filter (<70)", True)
        params['use_vol'] = st.checkbox("Volume Spike Filter", True)
        params['vol_mult'] = st.slider("Min Vol Spike (x)", 1.5, 5.0, 2.5)
        params['min_liq'] = 10000
    else:
        st.subheader("🏄 Trend Surfer Settings")
        
        use_adaptive = st.checkbox("🎯 Adaptive Stop-Loss (ATR)", value=False)
        params['use_adaptive_stop'] = use_adaptive
        
        if use_adaptive:
            params['atr_multiplier'] = st.slider("ATR Multiplier", 1.5, 4.0, 2.5, 0.5)
        else:
            params['stop_loss'] = st.slider("Fixed Stop Loss %", 3.0, 15.0, 7.0, 0.5)

# MAIN PAGE
st.markdown("---")

# REGIME INTELLIGENCE PANEL
with st.expander("🧠 Market Regime Intelligence", expanded=True):
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("🔄 Detect Current Regime", type="primary"):
            with st.spinner("Analyzing market conditions..."):
                nifty_current = fetch_nifty_data()
                if nifty_current is not None:
                    regime_type, confidence, description = detect_current_regime(nifty_current)
                    st.session_state['current_regime'] = {
                        'type': regime_type,
                        'confidence': confidence,
                        'description': description
                    }
    
    with col2:
        if 'current_regime' in st.session_state:
            regime_data = st.session_state['current_regime']
            regime_class = f"{regime_data['type'].lower()}-regime"
            
            st.markdown(f"""
                <div class="metric-box {regime_class}">
                    <h3>Current Regime: {regime_data['type']}</h3>
                    <p><b>Confidence:</b> {regime_data['confidence']}%</p>
                    <p>{regime_data['description']}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Strategy Recommendation
            rec = recommend_strategy(regime_data['type'], regime_data['confidence'])
            st.info(f"""
**📊 Recommended Strategy:** {rec['primary']}  
**Reason:** {rec['reason']}  
**Settings:** {rec['settings']}
            """)

st.markdown("---")

# SCANNER CONTROLS
with st.expander("🚀 Scanner & Backtest Controls", expanded=True):
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        market = st.radio("Market", ["NSE 500", "Commodities"], horizontal=True)
    
    with col2:
        backtest_period = st.selectbox(
            "Backtest Period",
            list(MARKET_PERIODS.keys()),
            index=len(MARKET_PERIODS) - 1  # Default to "Full Dataset"
        )
    
    with col3:
        run_btn = st.button("▶️ RUN", type="primary")

# Get selected period dates
period_info = MARKET_PERIODS[backtest_period]
start_date = period_info['start']
end_date = period_info['end']

st.caption(f"**Period:** {period_info['description']} | **Type:** {period_info['type']}")

# TABS
tab1, tab2, tab3 = st.tabs(["📡 Live Scanner", "📊 Bulk Backtest", "📈 Multi-Period Analysis"])

with tab1:
    if run_btn:
        t_list = NSE_500_LIST if market == "NSE 500" else list(COMMODITIES.values())
        
        with st.spinner(f"Running {strat_mode} on {backtest_period}..."):
            nifty_data = fetch_nifty_data(start_date, end_date)
            nifty_roc = nifty_data['ROC_55'] if nifty_data is not None else None
            
            results_buy, results_sell = run_master_scan(
                strat_mode, t_list, params, nifty_roc, start_date, end_date
            )
            
            st.session_state['scan_res_buy'] = results_buy
            st.session_state['scan_res_sell'] = results_sell
    
    c_buy, c_sell = st.columns(2)
    
    with c_buy:
        if 'scan_res_buy' in st.session_state:
            df_buy = pd.DataFrame(st.session_state['scan_res_buy'])
            st.success(f"✅ BUY SIGNALS ({len(df_buy)})")
            if not df_buy.empty:
                st.dataframe(df_buy, hide_index=True, use_container_width=True)
            else:
                st.info("No Buy Signals")
    
    with c_sell:
        if 'scan_res_sell' in st.session_state:
            df_sell = pd.DataFrame(st.session_state['scan_res_sell'])
            st.error(f"❌ SELL SIGNALS ({len(df_sell)})")
            if not df_sell.empty:
                st.dataframe(df_sell, hide_index=True, use_container_width=True)
            else:
                st.info("No Sell Signals")

with tab2:
    st.header(f"📊 {strat_mode}: {backtest_period}")
    
    col_filt1, col_filt2 = st.columns([1, 2])
    with col_filt1:
        min_profit = st.number_input("Min Profit % Filter", 0, 100, 5)
    
    if st.button("🚀 RUN BULK BACKTEST", type="primary"):
        t_list = NSE_500_LIST if market == "NSE 500" else list(COMMODITIES.values())
        
        with st.spinner("Crunching numbers..."):
            nifty_data = fetch_nifty_data(start_date, end_date)
            nifty_roc = nifty_data['ROC_55'] if nifty_data is not None else None
            
            prof, loss = run_bulk_test(
                strat_mode, t_list, params, nifty_roc, start_date, end_date
            )
            
            prof_clean = [p for p in prof if p['Raw_Ret'] >= min_profit]
            
            # Calculate metrics
            total_stocks = len(prof) + len(loss)
            win_rate = len(prof) / total_stocks * 100 if total_stocks > 0 else 0
            win_rate_clean = len(prof_clean) / total_stocks * 100 if total_stocks > 0 else 0
            
            # Display summary metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Stocks", total_stocks)
            m2.metric("Winners", f"{len(prof)} ({win_rate:.1f}%)")
            m3.metric(f"Winners >{min_profit}%", f"{len(prof_clean)} ({win_rate_clean:.1f}%)")
            m4.metric("Losers", f"{len(loss)} ({100-win_rate:.1f}%)")
            
            c1, c2 = st.columns(2)
            with c1:
                st.success(f"🏆 Profitable: {len(prof_clean)}")
                if prof_clean:
                    st.dataframe(pd.DataFrame(prof_clean).drop(columns=['Raw_Ret']), hide_index=True)
            
            with c2:
                st.error(f"⚠️ Loss Making: {len(loss)}")
                if loss:
                    st.dataframe(pd.DataFrame(loss).drop(columns=['Raw_Ret']).head(50), hide_index=True)

with tab3:
    st.header("📈 Cross-Period Performance Analysis")
    st.caption("Compare strategy performance across different market regimes")
    
    if st.button("🔬 Run Multi-Period Analysis", type="primary"):
        t_list = NSE_500_LIST if market == "NSE 500" else list(COMMODITIES.values())
        
        results_summary = []
        
        for period_name, period_data in MARKET_PERIODS.items():
            if period_name == "Full Dataset (2Y)":
                continue
            
            st.markdown(f"### Analyzing: {period_name}")
            
            with st.spinner(f"Testing {period_name}..."):
                nifty_data = fetch_nifty_data(period_data['start'], period_data['end'])
                nifty_roc = nifty_data['ROC_55'] if nifty_data is not None else None
                
                prof, loss = run_bulk_test(
                    strat_mode, t_list[:50], params, nifty_roc,  # Test on 50 stocks for speed
                    period_data['start'], period_data['end']
                )
                
                total = len(prof) + len(loss)
                win_rate = len(prof) / total * 100 if total > 0 else 0
                avg_return = np.mean([p['Raw_Ret'] for p in prof]) if prof else 0
                
                results_summary.append({
                    "Period": period_name,
                    "Type": period_data['type'],
                    "Win Rate": f"{win_rate:.1f}%",
                    "Winners": len(prof),
                    "Losers": len(loss),
                    "Avg Return": f"{avg_return:.1f}%"
                })
        
        st.markdown("### 📊 Summary Table")
        st.dataframe(pd.DataFrame(results_summary), hide_index=True, use_container_width=True)
        
        st.markdown("""
        **Key Insights:**
        - BULL periods should show >60% win rate
        - CHOPPY periods typically show 40-50% win rate
        - BEAR periods: Consider avoiding or shorting
        """)

st.markdown("---")
st.caption("⚡ ProTrader v2.0 | Powered by Vectorized Backtesting Engine | Data: Yahoo Finance")
