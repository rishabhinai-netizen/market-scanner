import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="ProTrader - Twin Engine", 
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
        .neutral { background-color: #f8f9fa; color: #666; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ ProTrader: Twin-Engine System")

# --- 2. ASSET LISTS ---
COMMODITIES = {
    'Gold (Global)': 'GC=F', 'Silver (Global)': 'SI=F', 'Copper (Global)': 'HG=F',
    'Crude Oil': 'CL=F', 'Natural Gas': 'NG=F'
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

# --- 3. CORE ANALYTICS ENGINE (Shared) ---
def calculate_indicators(df):
    """
    Calculates ALL indicators needed for BOTH strategies in one pass.
    """
    # 1. Donchian (For Strategy 1)
    df['High_20'] = df['High'].rolling(20).max()
    df['Low_20'] = df['Low'].rolling(20).min()
    df['Middle'] = (df['High_20'] + df['Low_20']) / 2
    
    # 2. Trend & MA (Shared)
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['SMA_5'] = df['Close'].rolling(5).mean() # For Mean Reversion Exit
    
    # 3. RSI 14 (For Strategy 1)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 4. RSI 2 (For Strategy 2 - Mean Reversion)
    gain_2 = (delta.where(delta > 0, 0)).rolling(2).mean()
    loss_2 = (-delta.where(delta < 0, 0)).rolling(2).mean()
    rs_2 = gain_2 / loss_2
    df['RSI_2'] = 100 - (100 / (1 + rs_2))
    
    # 5. Volume Stats (For Strategy 1)
    df['Vol_SMA_7'] = df['Volume'].rolling(7).mean().shift(1)
    
    return df

# --- 4. BACKTEST ENGINES (Strategy Specific) ---

def get_dc_stats(df, use_vol=False, vol_multiplier=2.5, min_liquidity=0):
    """STRATEGY 1: Donchian Breakout Backtest"""
    prev_close = df['Close'].shift(1)
    prev_mid = df['Middle'].shift(1)
    
    # Entry: Cross Up
    cross_up = (df['Close'] > df['Middle']) & (prev_close < prev_mid)
    # Exit: Cross Down
    cross_down = (df['Close'] < df['Middle']) & (prev_close > prev_mid)
    
    entry_signal = cross_up.copy()
    if use_vol:
        vol_cond = (df['Volume'] > vol_multiplier * df['Vol_SMA_7']) & (df['Vol_SMA_7'] > min_liquidity)
        entry_signal = entry_signal & vol_cond

    actions = pd.Series(np.nan, index=df.index)
    actions.loc[entry_signal] = 1 
    actions.loc[cross_down] = 0   
    
    df['Signal'] = actions.ffill().fillna(0).shift(1)
    
    df['Strat_Ret'] = df['Close'].pct_change() * df['Signal']
    cum_ret = (1 + df['Strat_Ret']).cumprod().iloc[-1] - 1
    trades = df['Signal'].diff().abs().sum() / 2
    
    return round(cum_ret * 100, 2), int(trades)

def get_mr_stats(df, rsi_entry=10):
    """STRATEGY 2: Mean Reversion (Dip Snap) Backtest"""
    # Logic: 
    # Buy IF: Close > SMA 200 (Uptrend) AND RSI_2 < Threshold (Panic)
    # Sell IF: Close > SMA 5 (Snap back)
    
    # 1. Define Conditions
    is_uptrend = df['Close'] > df['SMA_200']
    is_panic = df['RSI_2'] < rsi_entry
    
    entry_signal = is_uptrend & is_panic
    exit_signal = df['Close'] > df['SMA_5']
    
    # 2. Vectorized Latch (State Machine)
    actions = pd.Series(np.nan, index=df.index)
    actions.loc[entry_signal] = 1 # Enter
    actions.loc[exit_signal] = 0  # Exit
    
    # Fill NaN: If no signal today, keep previous state (Hold or Flat)
    df['Signal'] = actions.ffill().fillna(0).shift(1) # Enter next day
    
    # 3. Calc Returns
    df['Strat_Ret'] = df['Close'].pct_change() * df['Signal']
    cum_ret = (1 + df['Strat_Ret']).cumprod().iloc[-1] - 1
    trades = df['Signal'].diff().abs().sum() / 2
    
    return round(cum_ret * 100, 2), int(trades)

def process_batch(tickers, period="2y"):
    try:
        data = yf.download(tickers, period=period, group_by='ticker', threads=True, progress=False, auto_adjust=True)
        return data
    except: return None

# --- 5. MASTER SCANNER ---
@st.cache_data(ttl=900)
def run_master_scan(strategy_type, target_list, params):
    results = []
    
    chunk_size = 30
    total_chunks = range(0, len(target_list), chunk_size)
    progress = st.progress(0)
    status = st.empty()
    
    for i, start_idx in enumerate(total_chunks):
        batch = target_list[start_idx : start_idx + chunk_size]
        status.caption(f"Scanning batch {i+1}/{len(total_chunks)}...")
        
        data = process_batch(batch, period="2y")
        if data is None or data.empty: continue
        
        for ticker in batch:
            try:
                df = data[ticker] if len(batch) > 1 else data.copy()
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df = df.dropna()
                if len(df) < 200: continue # Need 200 SMA
                
                df = calculate_indicators(df)
                today = df.iloc[-1]
                prev = df.iloc[-2]
                display_name = ticker.replace('.NS', '').replace('=F', '')
                
                # --- STRATEGY BRANCHING ---
                
                if strategy_type == "Donchian Breakout":
                    # Check Live Entry
                    is_cross = (prev['Close'] < prev['Middle']) and (today['Close'] > today['Middle'])
                    if is_cross:
                        # Filters
                        if params['trend'] and today['Close'] < today['SMA_200']: continue
                        if params['rsi'] and today['RSI'] > 70: continue
                        
                        vol_spike = 0.0
                        if today['Vol_SMA_7'] > 0: vol_spike = today['Volume'] / today['Vol_SMA_7']
                        
                        if params['use_vol']:
                            if today['Vol_SMA_7'] < params['min_liq']: continue
                            if vol_spike < params['vol_mult']: continue
                        
                        # Backtest
                        ret, trades = get_dc_stats(df, params['use_vol'], params['vol_mult'], params['min_liq'])
                        
                        results.append({
                            "Symbol": display_name, "Price": round(today['Close'], 2),
                            "Signal": "Breakout", "Vol Spike": f"{round(vol_spike, 1)}x",
                            "Trades (2Y)": trades, "Ret (2Y)": f"{ret}%", "raw_ret": ret
                        })

                elif strategy_type == "Mean Reversion":
                    # Check Live Entry: Uptrend + Panic RSI
                    is_uptrend = today['Close'] > today['SMA_200']
                    is_panic = today['RSI_2'] < params['rsi_entry']
                    
                    if is_uptrend and is_panic:
                        # Backtest
                        ret, trades = get_mr_stats(df, params['rsi_entry'])
                        
                        results.append({
                            "Symbol": display_name, "Price": round(today['Close'], 2),
                            "Signal": "Dip Buy", "RSI(2)": round(today['RSI_2'], 1),
                            "Trades (2Y)": trades, "Ret (2Y)": f"{ret}%", "raw_ret": ret
                        })
                        
            except: continue
        progress.progress((i + 1) / len(total_chunks))
    
    if results:
        results.sort(key=lambda x: x['raw_ret'], reverse=True)
        for item in results: del item['raw_ret']
        
    progress.empty()
    status.empty()
    return results

@st.cache_data(ttl=3600)
def run_bulk_test(strategy_type, target_list, params):
    prof, loss = [], []
    chunk_size = 50
    chunks = range(0, len(target_list), chunk_size)
    progress = st.progress(0)
    
    for i, start in enumerate(chunks):
        batch = target_list[start : start + chunk_size]
        data = process_batch(batch)
        if data is None or data.empty: continue
        
        for ticker in batch:
            try:
                df = data[ticker] if len(batch) > 1 else data.copy()
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df = df.dropna()
                if len(df) < 200: continue
                df = calculate_indicators(df)
                
                if strategy_type == "Donchian Breakout":
                    ret, trades = get_dc_stats(df, params['use_vol'], params['vol_mult'], params['min_liq'])
                else:
                    ret, trades = get_mr_stats(df, params['rsi_entry'])
                
                entry = {
                    "Symbol": ticker.replace('.NS', ''), "Total Return": f"{ret}%",
                    "Trades": trades, "Raw_Ret": ret
                }
                
                if ret > 0: prof.append(entry)
                else: loss.append(entry)
            except: continue
        progress.progress((i+1)/len(chunks))
        
    progress.empty()
    prof.sort(key=lambda x: x['Raw_Ret'], reverse=True)
    loss.sort(key=lambda x: x['Raw_Ret'])
    return prof, loss

def deep_dive_chart(ticker, strategy_type, params):
    try:
        search_t = ticker
        rev_map = {k.split()[0].upper(): v for k, v in COMMODITIES.items()}
        if search_t in rev_map: search_t = rev_map[search_t]
        elif not search_t.endswith(".NS") and "=" not in search_t: search_t += ".NS"
        
        df = yf.download(search_t, period="2y", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = calculate_indicators(df)
        curr = df.iloc[-1]
        
        res = {
            "name": ticker.replace('.NS', ''), "cmp": round(curr['Close'], 2),
            "status": "Neutral", "box_color": "neutral"
        }

        if strategy_type == "Donchian Breakout":
            ret, trades = get_dc_stats(df, params['use_vol'], params['vol_mult'])
            
            # Live Status Logic
            is_bullish = curr['Close'] > curr['Middle']
            if is_bullish:
                res["status"] = "BULLISH (Above Mid Band)"
                res["box_color"] = "buy-signal"
            
            res.update({"ret": ret, "trades": trades, "info": f"Vol Spike: {round(curr['Volume']/curr['Vol_SMA_7'], 1)}x" if curr['Vol_SMA_7'] > 0 else "N/A"})
            
        else: # Mean Reversion
            ret, trades = get_mr_stats(df, params['rsi_entry'])
            
            # Live Status Logic
            in_trade = (curr['Close'] > curr['SMA_200']) and (curr['Close'] < curr['SMA_5']) # Holding
            can_buy = (curr['Close'] > curr['SMA_200']) and (curr['RSI_2'] < params['rsi_entry'])
            
            if can_buy:
                res["status"] = "BUY SIGNAL (Panic Dip)"
                res["box_color"] = "buy-signal"
            elif in_trade:
                res["status"] = "HOLD (Waiting for Snap)"
                res["box_color"] = "neutral"
            else:
                res["status"] = "NO SIGNAL"
                
            res.update({"ret": ret, "trades": trades, "info": f"RSI(2): {round(curr['RSI_2'], 1)}"})

        return res
    except: return None

# --- 6. UI LAYOUT ---

# SIDEBAR: Strategy Selector
with st.sidebar:
    st.header("🎮 Strategy Control")
    strat_mode = st.radio("Select Engine", ["Donchian Breakout", "Mean Reversion"], 
                          help="Donchian: Trend Following | Mean Reversion: Dip Buying")
    
    params = {}
    st.divider()
    
    if strat_mode == "Donchian Breakout":
        st.subheader("🌊 Breakout Settings")
        params['trend'] = st.checkbox("Trend Filter (>200 SMA)", True)
        params['rsi'] = st.checkbox("RSI Filter (<70)", True)
        st.divider()
        params['use_vol'] = st.checkbox("Volume Spike Filter", True)
        params['vol_mult'] = st.slider("Min Vol Spike (x)", 1.5, 5.0, 2.5)
        params['min_liq'] = 10000
        st.info("Best for: Strong Bull Markets")
        
    else: # Mean Reversion
        st.subheader("🧲 Mean Reversion Settings")
        st.caption("Strategy: Buy Uptrend Dips, Sell Snaps")
        params['rsi_entry'] = st.slider("Entry: RSI(2) Below", 2, 25, 10, help="Lower = More Extreme Panic")
        st.info("Best for: Choppy/Volatile Markets")

# MAIN PAGE
with st.expander("🚀 Scanner Controls", expanded=True):
    c1, c2 = st.columns([2,1])
    with c1:
        market = st.radio("Market", ["NSE 500", "Commodities"], horizontal=True)
    with c2:
        run_btn = st.button("RUN SCANNER")

tab1, tab2, tab3 = st.tabs(["📡 Live Scanner", "🔍 Deep Dive", "📊 Bulk Backtest"])

with tab1:
    if run_btn:
        t_list = NSE_500_LIST if market == "NSE 500" else list(COMMODITIES.values())
        with st.spinner(f"Running {strat_mode} Engine..."):
            results = run_master_scan(strat_mode, t_list, params)
            st.session_state['scan_res'] = results
            
    if 'scan_res' in st.session_state:
        df_res = pd.DataFrame(st.session_state['scan_res'])
        if not df_res.empty:
            st.success(f"✅ FOUND {len(df_res)} SIGNALS")
            st.caption("Sorted by 2-Year Historical Return")
            
            event = st.dataframe(
                df_res, hide_index=True, use_container_width=True,
                on_select="rerun", selection_mode="single-row"
            )
            if event.selection.rows:
                sel_ticker = df_res.iloc[event.selection.rows[0]]['Symbol']
                st.session_state['dd_ticker'] = sel_ticker
                st.info(f"Selected {sel_ticker} for Deep Dive.")
        else:
            st.warning("No stocks matched criteria today.")

with tab2:
    st.header("Deep Dive Analysis")
    default = st.session_state.get('dd_ticker', '')
    user_in = st.text_input("Symbol", value=default).upper().strip()
    
    if st.button("Analyze Stock") or user_in:
        if user_in:
            data = deep_dive_chart(user_in, strat_mode, params)
            if data:
                st.markdown(f"""
                    <div class="metric-box {data['box_color']}">
                        <h2>{data['name']}</h2>
                        <h3>CMP: {data['cmp']} | {data['status']}</h3>
                        <p>{data['info']} | <b>Trades (2Y):</b> {data['trades']} | <b>Total Ret (2Y):</b> {data['ret']}%</p>
                    </div>
                """, unsafe_allow_html=True)
            else: st.error("Not found.")

with tab3:
    st.header(f"📊 {strat_mode}: Historical Performance")
    st.write("Tests strategy on ALL stocks over last 2 years.")
    
    if st.button("RUN BULK BACKTEST", type="primary"):
        t_list = NSE_500_LIST if market == "NSE 500" else list(COMMODITIES.values())
        with st.spinner("Crunching Numbers..."):
            prof, loss = run_bulk_test(strat_mode, t_list, params)
            
            c1, c2 = st.columns(2)
            with c1:
                st.success(f"🏆 Profitable: {len(prof)}")
                if prof: st.dataframe(pd.DataFrame(prof).drop(columns=['Raw_Ret']), hide_index=True)
            with c2:
                st.error(f"⚠️ Loss Making: {len(loss)}")
                if loss: st.dataframe(pd.DataFrame(loss).drop(columns=['Raw_Ret']), hide_index=True)
