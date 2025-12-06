import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Donchian Pro", layout="wide", initial_sidebar_state="expanded")

# CSS: Hides ONLY the footer and deploy button. Keeps Header/Sidebar visible.
st.markdown("""
    <style>
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
        
        /* Box Styling */
        .metric-box {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            text-align: center;
            font-weight: bold;
        }
        .bullish { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .bearish { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .neutral { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Donchian Trend Sniper")

# --- 2. ASSET LISTS ---
COMMODITIES = {
    'Gold (USD)': 'GC=F', 'Silver (USD)': 'SI=F', 'Copper (USD)': 'HG=F',
    'Crude Oil (USD)': 'CL=F', 'Natural Gas (USD)': 'NG=F'
}

# FULL NIFTY 500 LIST (Hardcoded for reliability)
NSE_500_LIST = [
    'RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS', 'ITC.NS', 'LT.NS', 'BHARTIARTL.NS',
    'AXISBANK.NS', 'SBIN.NS', 'BAJFINANCE.NS', 'KOTAKBANK.NS', 'HINDUNILVR.NS', 'MARUTI.NS', 'TITAN.NS',
    'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'TATAMOTORS.NS', 'ADANIENT.NS', 'M&M.NS', 'NTPC.NS', 'POWERGRID.NS',
    'TATASTEEL.NS', 'JSWSTEEL.NS', 'HCLTECH.NS', 'COALINDIA.NS', 'ADANIPORTS.NS', 'ONGC.NS', 'GRASIM.NS',
    'BAJAJFINSV.NS', 'WIPRO.NS', 'DRREDDY.NS', 'CIPLA.NS', 'HINDALCO.NS', 'SBILIFE.NS', 'TECHM.NS',
    'BRITANNIA.NS', 'BPCL.NS', 'EICHERMOT.NS', 'INDUSINDBK.NS', 'HEROMOTOCO.NS', 'DIVISLAB.NS', 'APOLLOHOSP.NS',
    'TATACONSUM.NS', 'ASIANPAINT.NS', 'BEL.NS', 'HAL.NS', 'VBL.NS', 'TRENT.NS', 'SIEMENS.NS', 'ABB.NS',
    'PFC.NS', 'RECLTD.NS', 'IOC.NS', 'VEDANTA.NS', 'DLF.NS', 'ZOMATO.NS', 'JIOFIN.NS', 'GAIL.NS',
    'AMBUJACEM.NS', 'SHREECEM.NS', 'CHOLAFIN.NS', 'TVSMOTOR.NS', 'BANKBARODA.NS', 'CANBK.NS', 'PNB.NS',
    'UNIONBANK.NS', 'IDFCFIRSTB.NS', 'AUROPHARMA.NS', 'LUPIN.NS', 'ALKEM.NS', 'TORNTPHARM.NS', 'PIDILITIND.NS',
    'BERGEPAINT.NS', 'NAUKRI.NS', 'POLICYBZR.NS', 'IRCTC.NS', 'RVNL.NS', 'IRFC.NS', 'BHEL.NS', 'NHPC.NS',
    'SJVN.NS', 'SUZLON.NS', 'IDEA.NS', 'YESBANK.NS', 'GMRINFRA.NS', 'PRESTIGE.NS', 'OBEROIRLTY.NS',
    'GODREJPROP.NS', 'PHOENIXLTD.NS', 'ASHOKLEY.NS', 'BHARATFORG.NS', 'MRF.NS', 'BOSCHLTD.NS', 'PAGEIND.NS',
    'JUBLFOOD.NS', 'UBL.NS', 'MCDOWELL-N.NS', 'HINDCOPPER.NS', 'NATIONALUM.NS', 'SAIL.NS', 'JINDALSTEL.NS'
]

# --- 3. ANALYTICS ENGINE ---
def calculate_indicators(df):
    """Adds Donchian (20), SMA (200) and RSI"""
    # Donchian Channel (20)
    df['High_20'] = df['High'].rolling(20).max()
    df['Low_20'] = df['Low'].rolling(20).min()
    df['Middle'] = (df['High_20'] + df['Low_20']) / 2
    
    # Trend Filter (200 SMA)
    df['SMA_200'] = df['Close'].rolling(200).mean()
    
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

@st.cache_data(ttl=900)
def run_scan(tickers, use_trend=True, use_rsi=True):
    """Scans list for FRESH signals (Last 2 days)"""
    try:
        # Fetch 6mo data for scanning
        data = yf.download(tickers, period="6mo", group_by='ticker', threads=True, progress=False)
    except:
        return [], []
    
    buys, exits = [], []
    
    for ticker in tickers:
        try:
            # Data Extraction
            df = data[ticker] if len(tickers) > 1 else data
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            if len(df) < 50: continue
            
            df = calculate_indicators(df)
            
            today = df.iloc[-1]
            prev = df.iloc[-2]
            
            # --- SIGNAL LOGIC (FRESH ONLY) ---
            # Buy: Cross Over Middle Band
            if prev['Close'] < prev['Middle'] and today['Close'] > today['Middle']:
                # Apply Filters
                if use_trend and today['Close'] < today['SMA_200']: continue
                if use_rsi and today['RSI'] > 70: continue
                
                buys.append({
                    "Symbol": ticker,
                    "Price": round(today['Close'], 2),
                    "RSI": round(today['RSI'], 1),
                    "Trend": "⬆️ Uptrend" if today['Close'] > today['SMA_200'] else "⬇️ Weak"
                })
            
            # Exit: Cross Under Middle Band
            elif prev['Close'] > prev['Middle'] and today['Close'] < today['Middle']:
                exits.append({
                    "Symbol": ticker,
                    "Price": round(today['Close'], 2),
                    "RSI": round(today['RSI'], 1)
                })
        except: continue
        
    return buys, exits

def get_detailed_status(ticker):
    """Deep dive into a single stock to find historical entry"""
    try:
        # Fetch 2 Years of data to find start of trend
        df = yf.download(ticker, period="2y", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        df = calculate_indicators(df)
        curr = df.iloc[-1]
        
        # Determine Current Zone
        is_bullish = curr['Close'] > curr['Middle']
        status = "BUY ZONE" if is_bullish else "SELL/EXIT ZONE"
        
        # Find LAST Crossover Dates
        df['Buy_Cross'] = (df['Close'] > df['Middle']) & (df['Close'].shift(1) < df['Middle'].shift(1))
        df['Exit_Cross'] = (df['Close'] < df['Middle']) & (df['Close'].shift(1) > df['Middle'].shift(1))
        
        last_buy = df[df['Buy_Cross']].tail(1)
        last_exit = df[df['Exit_Cross']].tail(1)
        
        result = {
            "name": ticker,
            "cmp": round(curr['Close'], 2),
            "status": status,
            "is_bullish": is_bullish,
            "entry_date": "N/A", "entry_price": 0, "pnl": 0.0,
            "exit_date": "N/A", "exit_price": 0
        }
        
        if not last_buy.empty:
            result['entry_date'] = last_buy.index[-1].strftime('%d-%b-%Y')
            result['entry_price'] = round(last_buy['Close'].values[-1], 2)
            if is_bullish:
                result['pnl'] = round(((curr['Close'] - result['entry_price']) / result['entry_price']) * 100, 2)
                
        if not last_exit.empty:
            result['exit_date'] = last_exit.index[-1].strftime('%d-%b-%Y')
            result['exit_price'] = round(last_exit['Close'].values[-1], 2)
            
        return result
    except: return None

# --- 4. APP INTERFACE ---

# Sidebar Controls
st.sidebar.header("⚙️ Scanner Settings")
market_type = st.sidebar.radio("Market", ["NSE Nifty 500", "Commodities"])
st.sidebar.markdown("---")
st.sidebar.caption("Filters (For Buy Signals)")
trend_on = st.sidebar.checkbox("Trend Filter (200 SMA)", True)
rsi_on = st.sidebar.checkbox("RSI Filter (< 70)", True)

# Main Tab
tab1, tab2 = st.tabs(["🚀 Scanner Dashboard", "🔍 Individual Search"])

with tab1:
    if st.button("RUN SCANNER", type="primary"):
        t_list = NSE_500_LIST if market_type == "NSE Nifty 500" else list(COMMODITIES.values())
        
        with st.spinner("Scanning markets..."):
            buys, exits = run_scan(t_list, trend_on, rsi_on)
            
            # Save to session state to keep data after interaction
            st.session_state['buys'] = buys
            st.session_state['exits'] = exits

    # Display Results if they exist
    if 'buys' in st.session_state:
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader(f"✅ Fresh Buy Signals ({len(st.session_state['buys'])})")
            if st.session_state['buys']:
                df_buys = pd.DataFrame(st.session_state['buys'])
                
                # Interactive Table
                selection = st.dataframe(
                    df_buys, 
                    use_container_width=True, 
                    hide_index=True,
                    on_select="rerun",  # Makes rows clickable
                    selection_mode="single-row"
                )
                
                # Check if user clicked a row
                if selection.selection.rows:
                    idx = selection.selection.rows[0]
                    selected_ticker = df_buys.iloc[idx]['Symbol']
                    st.session_state['selected_stock'] = selected_ticker

        with c2:
            st.subheader(f"❌ Fresh Exit Signals ({len(st.session_state['exits'])})")
            if st.session_state['exits']:
                st.dataframe(pd.DataFrame(st.session_state['exits']), use_container_width=True, hide_index=True)

    # --- CLICK ACTION AREA ---
    if 'selected_stock' in st.session_state:
        st.markdown("---")
        st.header(f"📊 Analysis: {st.session_state['selected_stock']}")
        
        data = get_detailed_status(st.session_state['selected_stock'])
        
        if data:
            # Status Banner
            color_class = "bullish" if data['is_bullish'] else "bearish"
            st.markdown(f"""
                <div class="metric-box {color_class}">
                    <span style="font-size: 24px;">{data['status']}</span><br>
                    CMP: {data['cmp']}
                </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Last Buy Signal", f"{data['entry_date']}", f"@{data['entry_price']}")
            m2.metric("Last Exit Signal", f"{data['exit_date']}", f"@{data['exit_price']}")
            m3.metric("Current P&L (Run)", f"{data['pnl']}%", delta_color="normal")
            
            # Chart
            st.caption("Price vs Middle Band (20)")
            chart_df = yf.download(st.session_state['selected_stock'], period="1y", progress=False)
            if not chart_df.empty:
                if isinstance(chart_df.columns, pd.MultiIndex): chart_df.columns = chart_df.columns.get_level_values(0)
                # Recalculate middle for chart
                chart_df['High_20'] = chart_df['High'].rolling(20).max()
                chart_df['Low_20'] = chart_df['Low'].rolling(20).min()
                chart_df['Middle'] = (chart_df['High_20'] + chart_df['Low_20']) / 2
                st.line_chart(chart_df[['Close', 'Middle']])

with tab2:
    st.header("Search Stock History")
    search = st.text_input("Enter Symbol (e.g. TATASTEEL)", "").upper().strip()
    if st.button("Analyze"):
        if search:
            # Map Commodities
            t_search = search
            rev_map = {k.split()[0].upper(): v for k, v in COMMODITIES.items()}
            if t_search in rev_map: t_search = rev_map[t_search]
            elif not t_search.endswith(".NS") and "=" not in t_search: t_search += ".NS"
            
            # Reuse the same function
            data = get_detailed_status(t_search)
            if data:
                color_class = "bullish" if data['is_bullish'] else "bearish"
                st.markdown(f"""
                    <div class="metric-box {color_class}">
                        <span style="font-size: 24px;">{data['status']}</span><br>
                        CMP: {data['cmp']}
                    </div>
                """, unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Last Buy Signal", f"{data['entry_date']}", f"@{data['entry_price']}")
                c2.metric("Last Exit Signal", f"{data['exit_date']}", f"@{data['exit_price']}")
                c3.metric("Current P&L", f"{data['pnl']}%")
            else:
                st.error("Stock not found or Data unavailable.")
