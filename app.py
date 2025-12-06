import streamlit as st
import yfinance as yf
import pandas as pd
import time

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Donchian Sniper Pro", layout="wide", initial_sidebar_state="expanded")

# CSS: Professional Styling & Privacy (Hides Footer/Deploy)
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display:none;}
        
        div.stButton > button { background-color: #00C853; color: white; font-weight: bold; border: none; width: 100%; }
        
        .metric-box { padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 10px; border: 1px solid #ddd; }
        .buy-signal { background-color: #d4edda; color: #155724; border-color: #c3e6cb; }
        .sell-signal { background-color: #f8d7da; color: #721c24; border-color: #f5c6cb; }
        .neutral { background-color: #f8f9fa; color: #666; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Donchian Trend Sniper (Verified Nifty 500)")

# --- 2. ASSET LISTS ---
COMMODITIES = {
    'Gold (Global)': 'GC=F', 'Silver (Global)': 'SI=F', 'Copper (Global)': 'HG=F',
    'Crude Oil': 'CL=F', 'Natural Gas': 'NG=F', 'Aluminum': 'ALI=F'
}

# COMPLETE NIFTY 500 LIST (Verified from MW-NIFTY-500-06-Dec-2025.csv)
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

# --- 3. DATA & STRATEGY ENGINE ---
def process_batch(tickers):
    """Downloads batch with AUTO-ADJUST to ensure Price Accuracy"""
    try:
        # period="3mo" is enough for 20-day channels + buffers
        data = yf.download(tickers, period="3mo", group_by='ticker', threads=True, progress=False, auto_adjust=True)
        return data
    except Exception:
        return None

def calculate_donchian_signal(df):
    """
    STRICT STRATEGY LOGIC:
    BUY: Yesterday Close < Middle AND Today Close > Middle
    SELL: Yesterday Close > Middle AND Today Close < Middle
    """
    if len(df) < 22: return None, 0, None

    # 1. Donchian Calculation (20 Period)
    df['High_20'] = df['High'].rolling(window=20).max()
    df['Low_20'] = df['Low'].rolling(window=20).min()
    df['Middle'] = (df['High_20'] + df['Low_20']) / 2
    
    # 2. Extract Last Two Completed Candles
    today = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 3. Apply Signal Rules
    if prev['Close'] < prev['Middle'] and today['Close'] > today['Middle']:
        return "BUY", today['Close'], today.name

    elif prev['Close'] > prev['Middle'] and today['Close'] < today['Middle']:
        return "SELL", today['Close'], today.name
        
    return None, today['Close'], today.name

# --- 4. SCANNER FUNCTION ---
@st.cache_data(ttl=600)
def run_robust_scan(target_list):
    results_buy = []
    results_sell = []
    
    # Process in batches of 30 to prevent Data Corruption/Timeouts
    chunk_size = 30
    total_chunks = range(0, len(target_list), chunk_size)
    
    progress = st.progress(0)
    status_text = st.empty()
    
    for i, start_idx in enumerate(total_chunks):
        batch = target_list[start_idx : start_idx + chunk_size]
        status_text.caption(f"Scanning batch {i+1}/{len(total_chunks)}...")
        
        data = process_batch(batch)
        if data is None or data.empty: continue
        
        for ticker in batch:
            try:
                # Handle Multi-Index cleanly
                if len(batch) > 1:
                    df = data[ticker].copy()
                else:
                    df = data.copy()
                
                df.dropna(inplace=True)
                if df.empty: continue
                
                signal, price, date = calculate_donchian_signal(df)
                
                # CLEAN DISPLAY NAME (Remove .NS and =F)
                display_name = ticker.replace('.NS', '').replace('=F', '')
                
                if signal == "BUY":
                    results_buy.append({
                        "Symbol": display_name,
                        "Price": round(price, 2),
                        "Signal Date": date.strftime('%Y-%m-%d')
                    })
                elif signal == "SELL":
                    results_sell.append({
                        "Symbol": display_name,
                        "Price": round(price, 2),
                        "Signal Date": date.strftime('%Y-%m-%d')
                    })
                    
            except Exception:
                continue 
        
        progress.progress((i + 1) / len(total_chunks))
    
    progress.empty()
    status_text.empty()
    return results_buy, results_sell

def deep_dive_analysis(ticker):
    """Historical Analysis for Single Stock"""
    try:
        # Search Mapping
        search_t = ticker
        rev_map = {k.split()[0].upper(): v for k, v in COMMODITIES.items()}
        if search_t in rev_map: search_t = rev_map[search_t]
        elif not search_t.endswith(".NS") and "=" not in search_t: search_t += ".NS"

        # Fetch 1 Year Data
        df = yf.download(search_t, period="1y", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if len(df) < 20: return None, None

        # Strategy
        df['High_20'] = df['High'].rolling(20).max()
        df['Low_20'] = df['Low'].rolling(20).min()
        df['Middle'] = (df['High_20'] + df['Low_20']) / 2
        
        curr = df.iloc[-1]
        
        # Trend Status
        is_bullish = curr['Close'] > curr['Middle']
        trend_txt = "BULLISH (Buy Zone)" if is_bullish else "BEARISH (Sell Zone)"
        
        # Historical Signals
        df['Buy_Cross'] = (df['Close'] > df['Middle']) & (df['Close'].shift(1) < df['Middle'].shift(1))
        df['Sell_Cross'] = (df['Close'] < df['Middle']) & (df['Close'].shift(1) > df['Middle'].shift(1))
        
        last_buy = df[df['Buy_Cross']].tail(1)
        last_sell = df[df['Sell_Cross']].tail(1)
        
        res = {
            "name": ticker.replace('.NS', ''),
            "cmp": round(curr['Close'], 2),
            "trend": trend_txt,
            "is_bullish": is_bullish,
            "buy_date": "-", "buy_price": 0,
            "sell_date": "-", "sell_price": 0,
            "pnl": 0.0
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
    except:
        return None, None

# --- 5. UI LAYOUT ---
tab1, tab2 = st.tabs(["🚀 Market Scanner", "🔍 Deep Dive Analysis"])

with tab1:
    col1, col2 = st.columns([1, 4])
    with col1:
        st.subheader("Config")
        market = st.radio("Market", ["NSE 500 (Full)", "Commodities"])
        run_btn = st.button("RUN SCANNER", type="primary")
        
    with col2:
        if run_btn:
            scan_list = NSE_500_LIST if market == "NSE 500 (Full)" else list(COMMODITIES.values())
            st.toast(f"Starting Scan on {len(scan_list)} assets...")
            buys, sells = run_robust_scan(scan_list)
            
            st.session_state['scan_buys'] = buys
            st.session_state['scan_sells'] = sells
            
        # Results Display
        if 'scan_buys' in st.session_state:
            c1, c2 = st.columns(2)
            with c1:
                st.success(f"✅ BUY SIGNALS ({len(st.session_state['scan_buys'])})")
                if st.session_state['scan_buys']:
                    df_b = pd.DataFrame(st.session_state['scan_buys'])
                    st.dataframe(df_b, hide_index=True, use_container_width=True)
                    
                    # One-Click Analyze
                    selected_buy = st.selectbox("Analyze a Buy Signal:", ["Select..."] + df_b['Symbol'].tolist())
                    if selected_buy != "Select...":
                        st.session_state['analyze_ticker'] = selected_buy

            with c2:
                st.error(f"❌ SELL SIGNALS ({len(st.session_state['scan_sells'])})")
                if st.session_state['scan_sells']:
                    df_s = pd.DataFrame(st.session_state['scan_sells'])
                    st.dataframe(df_s, hide_index=True, use_container_width=True)

with tab2:
    st.header("Deep Dive Analysis")
    
    # Input Handling
    default_val = st.session_state.get('analyze_ticker', '')
    user_input = st.text_input("Enter Symbol (e.g. RELIANCE, GOLD)", value=default_val).upper().strip()
    
    if st.button("ANALYZE STOCK") or user_input:
        if user_input:
            data, chart_df = deep_dive_analysis(user_input)
            
            if data:
                # Dynamic Banner
                color = "buy-signal" if data['is_bullish'] else "sell-signal"
                st.markdown(f"""
                    <div class="metric-box {color}">
                        <h2>{data['name']}</h2>
                        <h3>CMP: {data['cmp']} | {data['trend']}</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Metrics Grid
                m1, m2, m3 = st.columns(3)
                m1.metric("Last Buy Signal", f"{data['buy_date']}", f"@{data['buy_price']}")
                m2.metric("Last Sell Signal", f"{data['sell_date']}", f"@{data['sell_price']}")
                m3.metric("Current P&L", f"{data['pnl']}%", delta_color="normal")
                
                # Chart
                if chart_df is not None:
                    st.line_chart(chart_df[['Close', 'Middle']])
            else:
                st.error("Data not found. Please check symbol.")
