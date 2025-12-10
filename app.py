import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="DC - Created by Rishabh", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# CSS: Professional UI & Mobile Optimizations
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display:none;}
        
        /* Button Styling */
        div.stButton > button { 
            background-color: #00C853; 
            color: white; 
            font-weight: bold; 
            border: none; 
            width: 100%; 
        }
        
        /* Metric Box Styling */
        .metric-box { 
            padding: 15px; 
            border-radius: 8px; 
            text-align: center; 
            margin-bottom: 10px; 
            border: 1px solid #ddd; 
        }
        .buy-signal { background-color: #d4edda; color: #155724; border-color: #c3e6cb; }
        .sell-signal { background-color: #f8d7da; color: #721c24; border-color: #f5c6cb; }
        .neutral { background-color: #f8f9fa; color: #666; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ DC - Created by Rishabh")

# --- 2. ASSET LISTS ---
COMMODITIES = {
    'Gold (Global)': 'GC=F', 'Silver (Global)': 'SI=F', 'Copper (Global)': 'HG=F',
    'Crude Oil': 'CL=F', 'Natural Gas': 'NG=F', 'Aluminum': 'ALI=F'
}

NSE_500_LIST = [
    'M&MFIN.NS', 'RCF.NS', 'PATANJALI.NS', 'SBICARD.NS', 'SUNDARMFIN.NS', 'PTCIL.NS', 'INDUSTOWER.NS', 'CHOLAFIN.NS', 'LTF.NS', 'SKFINDUS.NS',
    'SHRIRAMFIN.NS', 'MARICO.NS', 'DEEPAKNTR.NS', 'ABCAPITAL.NS', 'SBIN.NS', 'PNBHOUSING.NS', 'MUTHOOTFIN.NS', 'RBLBANK.NS', 'POLICYBZR.NS', 'LLOYDSME.NS',
    'LINDEINDIA.NS', 'MCX.NS', 'BAJAJFINSV.NS', 'SUZLON.NS', 'ADANIENT.NS', 'MANAPPURAM.NS', 'HSCL.NS', 'PRESTIGE.NS', 'MARUTI.NS', 'CHOLAHLDNG.NS',
    'ELGIEQUIP.NS', 'BSE.NS', 'PNB.NS', 'ATUL.NS', 'ICICIPRULI.NS', 'BAJFINANCE.NS', 'GODREJIND.NS', 'SRF.NS', 'LEMONTREE.NS', 'EMAMILTD.NS',
    'VGUARD.NS', '360ONE.NS', 'HCLTECH.NS', 'UNITDSPR.NS', 'DLF.NS', 'PAYTM.NS', 'BRITANNIA.NS', 'SCI.NS', 'AWL.NS', 'NATIONALUM.NS',
    'MAXHEALTH.NS', 'EICHERMOT.NS', 'HINDALCO.NS', 'NUVAMA.NS', 'RAILTEL.NS', 'LT.NS', 'SIGNATURE.NS', 'GLAXO.NS', 'AIIL.NS', 'BANKBARODA.NS',
    'MPHASIS.NS', 'VENTIVE.NS', 'JSWSTEEL.NS', 'AUBANK.NS', 'LAURUSLABS.NS', 'BPCL.NS', 'WOCKPHARMA.NS', 'IDFCFIRSTB.NS', 'SAILIFE.NS', 'ASTRAL.NS',
    'HUDCO.NS', 'ABFRL.NS', 'TATACONSUM.NS', 'INDIAMART.NS', 'CDSL.NS', 'ONESOURCE.NS', 'COLPAL.NS', 'TRITURBINE.NS', 'HDFCLIFE.NS', 'AAVAS.NS',
    'DMART.NS', 'BANKINDIA.NS', 'PERSISTENT.NS', 'INFY.NS', 'JSL.NS', 'DELHIVERY.NS', 'M&M.NS', 'WIPRO.NS', 'KARURVYSYA.NS', 'SAREGAMA.NS',
    'OIL.NS', 'KOTAKBANK.NS', 'BIOCON.NS', 'IDEA.NS', 'FLUOROCHEM.NS', 'ANGELONE.NS', 'ADANIPOWER.NS', 'CANFINHOME.NS', 'BATAINDIA.NS', 'CRISIL.NS',
    'CRAFTSMAN.NS', 'GLAND.NS', 'FACT.NS', 'CENTURYPLY.NS', 'BALRAMCHIN.NS', 'SBILIFE.NS', 'HINDCOPPER.NS', 'INDUSINDBK.NS', 'CANBK.NS', 'BHEL.NS',
    'GMRAIRPORT.NS', 'SCHAEFFLER.NS', 'ADANIENSOL.NS', 'JSWENERGY.NS', 'SUPREMEIND.NS', 'INDIANB.NS', 'BHARATFORG.NS', 'BAJAJHLDNG.NS', 'GRASIM.NS', 'RELIANCE.NS',
    'TVSMOTOR.NS', 'JSWINFRA.NS', 'JUBLPHARMA.NS', 'CUB.NS', 'NLCINDIA.NS', 'DIXON.NS', 'JIOFIN.NS', 'ZENTEC.NS', 'CGCL.NS', 'CONCORDBIO.NS',
    'MAHSCOOTER.NS', 'NMDC.NS', 'CROMPTON.NS', 'ARE&M.NS', 'LTIM.NS', 'KALYANKJIL.NS', 'DABUR.NS', 'UPL.NS', 'ADANIGREEN.NS', 'ALKEM.NS',
    'TECHM.NS', 'ICICIBANK.NS', 'UNIONBANK.NS', 'HDFCBANK.NS', 'JBCHEPHARM.NS', 'AJANTPHARM.NS', 'IOC.NS', 'HOMEFIRST.NS', 'CEATLTD.NS', 'ITC.NS',
    'GODREJCP.NS', 'ASTRAZEN.NS', 'TCS.NS', 'TATASTEEL.NS', 'ASHOKLEY.NS', 'APARINDS.NS', 'ADANIPORTS.NS', 'NESTLEIND.NS', 'BIKAJI.NS', 'COFORGE.NS',
    'IIFL.NS', 'AGARWALEYE.NS', 'RECLTD.NS', 'TITAN.NS', 'PFC.NS', 'POWERGRID.NS', 'JUBLFOOD.NS', 'BHARTIARTL.NS', 'HINDPETRO.NS', 'BAJAJ-AUTO.NS',
    'DIVISLAB.NS', 'ABBOTINDIA.NS', 'SBFC.NS', 'CGPOWER.NS', 'INOXWIND.NS', 'COALINDIA.NS', 'COCHINSHIP.NS', 'DALBHARAT.NS', 'ENDURANCE.NS', 'OFSS.NS',
    'SUNDRMFAST.NS', 'HAVELLS.NS', 'VTL.NS', 'INDHOTEL.NS', 'HINDZINC.NS', 'JINDALSTEL.NS', 'SAIL.NS', 'ENGINERSIN.NS', 'NHPC.NS', 'RAINBOW.NS',
    'NTPC.NS', 'FEDERALBNK.NS', 'IRCTC.NS', 'ERIS.NS', 'ASIANPAINT.NS', 'MFSL.NS', 'AXISBANK.NS', 'GODREJPROP.NS', 'ABB.NS', 'HONAUT.NS',
    'UNOMINDA.NS', 'MSUMI.NS', 'NTPCGREEN.NS', 'ATHERENERG.NS', 'MAHABANK.NS', 'CLEAN.NS', 'KPRMILL.NS', 'EIDPARRY.NS', 'RITES.NS', 'HEROMOTOCO.NS',
    'TATAELXSI.NS', 'BOSCHLTD.NS', 'ATGL.NS', 'INDGN.NS', 'SHYAMMETL.NS', 'ENRIN.NS', 'JKCEMENT.NS', 'GRAVITA.NS', 'JYOTHYLAB.NS', 'CUMMINSIND.NS',
    'PAGEIND.NS', 'IRFC.NS', 'CHALET.NS', 'CIPLA.NS', 'VBL.NS', 'MANKIND.NS', 'SCHNEIDER.NS', 'TITAGARH.NS', 'AADHARHFC.NS', 'APOLLOHOSP.NS',
    'EXIDEIND.NS', 'CHAMBLFERT.NS', 'BASF.NS', 'LUPIN.NS', 'BEL.NS', 'ULTRACEMCO.NS', 'IOB.NS', 'TATAPOWER.NS', 'LODHA.NS', 'PFIZER.NS',
    'DRREDDY.NS', 'CASTROLIND.NS', 'ONGC.NS', 'BLS.NS', 'RKFORGE.NS', 'KIMS.NS', 'APLLTD.NS', 'AKUMS.NS', 'KPIL.NS', 'AFCONS.NS',
    'ABSLAMC.NS', 'APLAPOLLO.NS', 'CENTRALBK.NS', 'CHENNPETRO.NS', 'IDBI.NS', 'ITI.NS', 'RVNL.NS', 'RADICO.NS', 'CAMPUS.NS', 'NYKAA.NS',
    'VOLTAS.NS', 'POONAWALLA.NS', 'NAVINFLUOR.NS', 'OBEROIRLTY.NS', 'INDIACEM.NS', 'SUMICHEM.NS', 'WELCORP.NS', 'MGL.NS', 'ZEEL.NS', 'PCBL.NS',
    'ZYDUSLIFE.NS', 'GODIGIT.NS', 'MRF.NS', 'PVRINOX.NS', 'AMBUJACEM.NS', 'FINCABLES.NS', 'POLYMED.NS', 'JPPOWER.NS', 'ZENSARTECH.NS', 'GAIL.NS',
    'EMCURE.NS', 'BBTC.NS', 'J&KBANK.NS', 'BHARTIHEXA.NS', 'BLUEDART.NS', 'FORTIS.NS', 'ICICIGI.NS', 'NBCC.NS', 'ESCORTS.NS', 'BALKRISIND.NS',
    'CONCOR.NS', 'PIIND.NS', 'KAJARIACER.NS', 'MAPMYINDIA.NS', 'DCMSHRIRAM.NS', 'AUROPHARMA.NS', 'ALOKINDS.NS', 'NAVA.NS', 'SOBHA.NS', 'PGHH.NS',
    'HBLENGINE.NS', 'PPLPHARMA.NS', 'JUBLINGREA.NS', 'KFINTECH.NS', 'TORNTPHARM.NS', 'PHOENIXLTD.NS', 'HEG.NS', 'UBL.NS', 'IFCI.NS', 'BERGEPAINT.NS',
    'RAMCOCEM.NS', 'TRIDENT.NS', 'LICHSGFIN.NS', 'KPITTECH.NS', 'NSLNISP.NS', 'GRANULES.NS', 'VMM.NS', 'UCOBANK.NS', 'GUJGASLTD.NS', 'LICI.NS',
    'NAUKRI.NS', 'SKFINDIA.NS', 'BAYERCROP.NS', 'HDFCAMC.NS', 'PIDILITIND.NS', 'YESBANK.NS', 'IPCALAB.NS', 'TMPV.NS', 'MOTHERSON.NS', 'BAJAJHFL.NS',
    'KEI.NS', 'JYOTICNC.NS', 'IRB.NS', 'SUNPHARMA.NS', 'CYIENT.NS', 'TATACOMM.NS', 'TORNTPOWER.NS', 'COROMANDEL.NS', 'MINDACORP.NS', 'SOLARINDS.NS',
    'MAHSEAMLES.NS', 'TECHNOE.NS', 'EIHOTEL.NS', 'GLENMARK.NS', 'AKZOINDIA.NS', 'GODREJAGRO.NS', 'THERMAX.NS', 'ANANDRATHI.NS', 'AFFLE.NS', 'SUNTV.NS',
    'ACC.NS', 'TRENT.NS', 'GSPL.NS', 'BSOFT.NS', 'NCC.NS', 'GICRE.NS', 'INOXINDIA.NS', 'TATAINVEST.NS', 'DOMS.NS', 'HONASA.NS',
    'MAZDOCK.NS', 'USHAMART.NS', 'SYNGENE.NS', 'IKS.NS', 'IGL.NS', 'ZFCVINDIA.NS', 'TATACHEM.NS', 'ELECON.NS', 'NUVOCO.NS', 'ITCHOTELS.NS',
    'FSL.NS', 'SAMMAANCAP.NS', 'LALPATHLAB.NS', 'CESC.NS', 'LATENTVIEW.NS', 'FINPIPE.NS', 'BLUESTARCO.NS', 'VEDL.NS', 'POLYCAB.NS', 'NEULANDLAB.NS',
    'CAPLIPOINT.NS', 'NH.NS', 'METROPOLIS.NS', 'BRIGADE.NS', 'HAPPSTMNDS.NS', 'KSB.NS', 'CERA.NS', 'HAL.NS', 'RPOWER.NS', 'RHIM.NS',
    'TRIVENI.NS', 'BDL.NS', 'ABREL.NS', 'TIINDIA.NS', 'APOLLOTYRE.NS', 'TATATECH.NS', 'SIEMENS.NS', 'SARDAEN.NS', 'INDIGO.NS', 'GESHIP.NS',
    'SAGILITY.NS', 'SONACOMS.NS', 'JINDALSAW.NS', 'SHREECEM.NS', 'PRAJIND.NS', 'TIMKEN.NS', 'NIVABUPA.NS', 'ETERNAL.NS', 'JMFINANCIL.NS', 'APTUS.NS',
    'ACE.NS', 'ALKYLAMINE.NS', 'GVT&D.NS', 'KEC.NS', 'GRAPHITE.NS', 'BLUEJET.NS', 'GPIL.NS', 'MANYAVAR.NS', 'OLAELEC.NS', 'AEGISVOPAK.NS',
    'SWANCORP.NS', 'FIVESTAR.NS', 'NATCOPHARM.NS', 'IEX.NS', 'STARHEALTH.NS', 'WELSPUNLIV.NS', 'CARBORUNIV.NS', 'SONATSOFTW.NS', 'IRCON.NS', 'DBREALTY.NS',
    'COHANCE.NS', 'OLECTRA.NS', 'TEJASNET.NS', 'HFCL.NS', 'MMTC.NS', '3MINDIA.NS', 'ECLERX.NS', 'THELEELA.NS', 'AARTIIND.NS', 'CAMS.NS',
    'GODFRYPHLP.NS', 'AIAENG.NS', 'SWIGGY.NS', 'SJVN.NS', 'CHOICEIN.NS', 'DEVYANI.NS', 'JKTYRE.NS', 'UTIAMC.NS', 'MRPL.NS', 'RRKABEL.NS',
    'TTML.NS', 'PETRONET.NS', 'JBMA.NS', 'ASAHIINDIA.NS', 'GILLETTE.NS', 'IREDA.NS', 'NAM-INDIA.NS', 'FORCEMOT.NS', 'DEEPAKFERT.NS', 'KIRLOSBROS.NS',
    'NIACL.NS', 'NEWGEN.NS', 'AEGISLOG.NS', 'ANANTRAJ.NS', 'MEDANTA.NS', 'ASTERDM.NS', 'AMBER.NS', 'HYUNDAI.NS', 'GMDCLTD.NS', 'JWL.NS',
    'IGIL.NS', 'TBOTEK.NS', 'CCL.NS', 'LTTS.NS', 'ACMESOLAR.NS', 'RELINFRA.NS', 'KIRLOSENG.NS', 'TARIL.NS', 'FIRSTCRY.NS', 'INTELLECT.NS',
    'ABLBL.NS', 'WHIRLPOOL.NS', 'GRSE.NS', 'MOTILALOFS.NS', 'BEML.NS', 'HINDUNILVR.NS', 'LTFOODS.NS', 'POWERINDIA.NS', 'REDINGTON.NS', 'WAAREEENER.NS',
    'CREDITACC.NS', 'SAPPHIRE.NS', 'BANDHANBNK.NS', 'PGEL.NS', 'PREMIERENE.NS', 'HEXT.NS', 'DATAPATTNS.NS', 'NETWEB.NS', 'SYRMA.NS', 'VIJAYA.NS',
    'KAYNES.NS'
]

# --- 3. ANALYTICS ENGINE ---
def calculate_indicators(df):
    """Calculates Donchian (20), SMA (200) and RSI (14)"""
    df['High_20'] = df['High'].rolling(20).max()
    df['Low_20'] = df['Low'].rolling(20).min()
    df['Middle'] = (df['High_20'] + df['Low_20']) / 2
    df['SMA_200'] = df['Close'].rolling(200).mean()
    
    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def get_strategy_stats(df):
    """
    Calculates 2-Year Backtest metrics on the fly.
    Returns: Total Return % (float), Trade Count (int)
    """
    # 1. Generate Signal: 1 (Hold), 0 (Flat)
    # Be in market if Price > Middle
    df['Signal_Strat'] = np.where(df['Close'] > df['Middle'], 1, 0)
    df['Signal_Strat'] = df['Signal_Strat'].shift(1) # Enter next day (or close-to-close sim)
    
    # 2. Calculate Returns
    df['Daily_Ret'] = df['Close'].pct_change()
    df['Strat_Ret'] = df['Daily_Ret'] * df['Signal_Strat']
    
    # 3. Cumulative Metrics
    cum_ret = (1 + df['Strat_Ret']).cumprod().iloc[-1] - 1
    total_ret_pct = round(cum_ret * 100, 2)
    
    # 4. Count Trades (transitions from 0 to 1 or 1 to 0)
    # We divide by 2 because a full trade is Buy + Sell
    trades = df['Signal_Strat'].diff().abs().sum() / 2
    
    return total_ret_pct, int(trades)

def process_batch(tickers, period="2y"):
    """Downloads batch with AUTO-ADJUST"""
    try:
        # Fixed to 2y to ensure we always have enough data for backtesting stats
        data = yf.download(tickers, period=period, group_by='ticker', threads=True, progress=False, auto_adjust=True)
        return data
    except Exception:
        return None

# --- 4. SCANNER LOGIC ---
@st.cache_data(ttl=900)
def run_scan(target_list, use_trend, use_rsi):
    results_buy = []
    results_sell = []
    
    chunk_size = 30
    total_chunks = range(0, len(target_list), chunk_size)
    progress = st.progress(0)
    status = st.empty()
    
    for i, start_idx in enumerate(total_chunks):
        batch = target_list[start_idx : start_idx + chunk_size]
        status.caption(f"Scanning & Backtesting batch {i+1}/{len(total_chunks)}...")
        
        # Download 2 years data to support on-the-fly backtesting
        data = process_batch(batch, period="2y")
        if data is None or data.empty: continue
        
        for ticker in batch:
            try:
                df = data[ticker] if len(batch) > 1 else data.copy()
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df = df.dropna()
                if len(df) < 50: continue
                
                df = calculate_indicators(df)
                today = df.iloc[-1]
                prev = df.iloc[-2]
                
                display_name = ticker.replace('.NS', '').replace('=F', '')
                
                # BUY: Prev < Mid AND Curr > Mid
                if prev['Close'] < prev['Middle'] and today['Close'] > today['Middle']:
                    if use_trend and today['Close'] < today['SMA_200']: continue
                    if use_rsi and today['RSI'] > 70: continue
                    
                    # Run Instant Backtest
                    ret_2y, trades_2y = get_strategy_stats(df)
                    
                    results_buy.append({
                        "Symbol": display_name,
                        "Price": round(today['Close'], 2),
                        "RSI": round(today['RSI'], 1),
                        "Trend": "⬆️ Uptrend" if today['Close'] > today['SMA_200'] else "⬇️ Weak",
                        # Backtest result - Raw values used for sorting, strings for display later if needed
                        "raw_ret": ret_2y, 
                        "Trades (2Y)": trades_2y,
                        "Total Ret (2Y Backtest)": f"{ret_2y}%" # Formatted for display
                    })
                
                # SELL: Prev > Mid AND Curr < Mid
                elif prev['Close'] > prev['Middle'] and today['Close'] < today['Middle']:
                    # Note: Backtest results REMOVED for Sell signals to avoid confusion
                    
                    results_sell.append({
                        "Symbol": display_name,
                        "Price": round(today['Close'], 2),
                        "Signal Date": today.name.strftime('%Y-%m-%d')
                    })
                    
            except: continue
            
        progress.progress((i + 1) / len(total_chunks))
    
    # Sorting Buy Signals by "Most Favorable"
    # Criteria: Ranking by Backtested Return (High to Low)
    if results_buy:
        results_buy.sort(key=lambda x: x['raw_ret'], reverse=True)
        # Remove raw_ret before display to keep UI clean
        for item in results_buy:
            del item['raw_ret']

    progress.empty()
    status.empty()
    return results_buy, results_sell

# --- 5. BACKTEST LOGIC (Bulk) ---
@st.cache_data(ttl=3600)
def run_bulk_backtest(target_list):
    """
    Scans 2 years of history for ALL stocks.
    Returns: Profitable List, Loss List
    """
    profitable = []
    loss_making = []
    
    chunk_size = 50
    total_chunks = range(0, len(target_list), chunk_size)
    progress = st.progress(0)
    status = st.empty()
    
    for i, start_idx in enumerate(total_chunks):
        batch = target_list[start_idx : start_idx + chunk_size]
        status.caption(f"Backtesting batch {i+1}/{len(total_chunks)}...")
        
        data = process_batch(batch, period="2y")
        if data is None or data.empty: continue
        
        for ticker in batch:
            try:
                df = data[ticker] if len(batch) > 1 else data.copy()
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df = df.dropna()
                if len(df) < 200: continue
                
                df = calculate_indicators(df)
                
                # Reuse helper function
                total_ret_pct, trades = get_strategy_stats(df)
                
                entry = {
                    "Symbol": ticker.replace('.NS', '').replace('=F', ''),
                    "Total Return": f"{total_ret_pct}%",
                    "Trades (Approx)": trades,
                    "Raw_Ret": total_ret_pct
                }
                
                if total_ret_pct > 0:
                    profitable.append(entry)
                else:
                    loss_making.append(entry)
                    
            except: continue
        
        progress.progress((i + 1) / len(total_chunks))
    
    progress.empty()
    status.empty()
    
    # Sort by performance
    profitable.sort(key=lambda x: x['Raw_Ret'], reverse=True)
    loss_making.sort(key=lambda x: x['Raw_Ret']) # Most negative first
    
    return profitable, loss_making

def deep_dive(ticker):
    """Historical Check"""
    try:
        search_t = ticker
        rev_map = {k.split()[0].upper(): v for k, v in COMMODITIES.items()}
        if search_t in rev_map: search_t = rev_map[search_t]
        elif not search_t.endswith(".NS") and "=" not in search_t: search_t += ".NS"
        
        df = yf.download(search_t, period="2y", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        df = calculate_indicators(df)
        curr = df.iloc[-1]
        
        # Strategy Metrics
        ret_2y, trades_2y = get_strategy_stats(df)
        
        is_bullish = curr['Close'] > curr['Middle']
        
        # Status Update: Only show "Buy Zone" if Price > Middle. Else show "No Trade Alerts".
        if is_bullish:
             status = "BULLISH (Active Buy Signal)"
             box_color = "buy-signal"
        else:
             status = "No Trade Alerts"
             box_color = "neutral"
        
        df['Buy_X'] = (df['Close'] > df['Middle']) & (df['Close'].shift(1) < df['Middle'].shift(1))
        df['Sell_X'] = (df['Close'] < df['Middle']) & (df['Close'].shift(1) > df['Middle'].shift(1))
        
        last_buy = df[df['Buy_X']].tail(1)
        last_sell = df[df['Sell_X']].tail(1)
        
        res = {
            "name": ticker.replace('.NS', ''),
            "cmp": round(curr['Close'], 2),
            "status": status,
            "box_color": box_color,
            "is_bullish": is_bullish,
            "buy_date": "-", "buy_price": 0, "pnl": 0.0,
            "sell_date": "-", "sell_price": 0,
            "ret_2y": ret_2y,
            "trades_2y": trades_2y
        }
        
        if not last_buy.empty:
            res['buy_date'] = last_buy.index[-1].strftime('%d-%b-%Y')
            res['buy_price'] = round(last_buy['Close'].values[-1], 2)
            if is_bullish:
                res['pnl'] = round(((curr['Close'] - res['buy_price']) / res['buy_price']) * 100, 2)
                
        if not last_sell.empty:
            res['sell_date'] = last_sell.index[-1].strftime('%d-%b-%Y')
            res['sell_price'] = round(last_sell['Close'].values[-1], 2)
            
        return res, df
    except: return None, None

# --- 6. UI LAYOUT ---
# Mobile Optimization: Controls moved to Main Page Expander
with st.expander("⚙️ Scanner Settings & Controls", expanded=True):
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        market = st.radio("Select Market", ["NSE 500 (Full)", "Commodities"])
    with c2:
        st.caption("Filters")
        trend_filter = st.checkbox("Trend Filter (>200 SMA)", True)
        rsi_filter = st.checkbox("RSI Filter (<70)", True)
    with c3:
        st.write("")
        st.write("")
        run_btn = st.button("RUN SCANNER", type="primary")

tab1, tab2, tab3 = st.tabs(["🚀 Market Scanner", "🔍 Deep Dive", "📊 Bulk Backtest"])

with tab1:
    if run_btn:
        t_list = NSE_500_LIST if market == "NSE 500 (Full)" else list(COMMODITIES.values())
        with st.spinner("Scanning & Calculating 2-Year Returns..."):
            buys, sells = run_scan(t_list, trend_filter, rsi_filter)
            st.session_state['buys'] = buys
            st.session_state['sells'] = sells
        
    if 'buys' in st.session_state:
        c1, c2 = st.columns(2)
        with c1:
            st.success(f"✅ BUY SIGNALS ({len(st.session_state['buys'])})")
            if st.session_state['buys']:
                df_b = pd.DataFrame(st.session_state['buys'])
                
                # Interactive Dataframe for selection
                st.caption("👇 Select a row below to analyze in Deep Dive")
                event = st.dataframe(
                    df_b, 
                    hide_index=True, 
                    use_container_width=True,
                    on_select="rerun", # Interactive selection
                    selection_mode="single-row"
                )
                
                # Handle Selection
                if event.selection.rows:
                    idx = event.selection.rows[0]
                    selected_ticker = df_b.iloc[idx]['Symbol']
                    st.session_state['analyze_ticker'] = selected_ticker
                    st.info(f"👉 Selected **{selected_ticker}** for Deep Dive. Go to 'Deep Dive' tab.")

        with c2:
            st.error(f"❌ SELL SIGNALS ({len(st.session_state['sells'])})")
            if st.session_state['sells']:
                st.dataframe(pd.DataFrame(st.session_state['sells']), hide_index=True, use_container_width=True)

with tab2:
    st.header("Deep Dive Analysis")
    default = st.session_state.get('analyze_ticker', '')
    user_in = st.text_input("Enter Symbol (e.g. RELIANCE)", value=default).upper().strip()
    
    if st.button("ANALYZE") or user_in:
        if user_in:
            data, chart = deep_dive(user_in)
            if data:
                st.markdown(f"""
                    <div class="metric-box {data['box_color']}">
                        <h2>{data['name']}</h2>
                        <h3>CMP: {data['cmp']} | {data['status']}</h3>
                        <p><b>Trades (2Y):</b> {data['trades_2y']} | <b>Total Ret (2Y Backtest):</b> {data['ret_2y']}%</p>
                    </div>
                """, unsafe_allow_html=True)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Last Buy Signal", f"{data['buy_date']}", f"@{data['buy_price']}")
                m2.metric("Last Sell Signal", f"{data['sell_date']}", f"@{data['sell_price']}")
                m3.metric("Current P&L", f"{data['pnl']}%")
                
                # Chart removed as requested
                st.info("Chart removed to focus on data.")
            else: st.error("Stock not found.")

with tab3:
    st.header("📊 Strategy Performance (2 Years)")
    st.write("Bulk scan of all 500 stocks to find historical winners.")
    
    backtest_market = st.radio("Select Market for Backtest", ["NSE 500 (Full)", "Commodities"], key="bt_market")
    
    if st.button("RUN BULK BACKTEST", type="primary"):
        t_list = NSE_500_LIST if backtest_market == "NSE 500 (Full)" else list(COMMODITIES.values())
        
        with st.spinner("Crunching 2 years of data for all stocks... (60-90s)"):
            prof, loss = run_bulk_backtest(t_list)
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.success(f"🏆 Profitable Stocks ({len(prof)})")
                st.caption("Stocks where this strategy made money")
                if prof:
                    st.dataframe(pd.DataFrame(prof).drop(columns=['Raw_Ret']), use_container_width=True, hide_index=True)
            
            with col_b:
                st.error(f"⚠️ Loss Making Stocks ({len(loss)})")
                st.caption("Stocks where this strategy lost money")
                if loss:
                    st.dataframe(pd.DataFrame(loss).drop(columns=['Raw_Ret']), use_container_width=True, hide_index=True)
