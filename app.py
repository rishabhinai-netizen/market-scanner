import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# --- 1. CONFIGURATION & PRIVACY ---
st.set_page_config(page_title="Pro Market Scanner", layout="wide", initial_sidebar_state="expanded")

# CSS to Hide Streamlit Toolbar, Footer, and Deploy Button
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display:none;}
        
        /* Custom UI Styling */
        div.stButton > button { background-color: #00C853; color: white; font-weight: bold; border: none; width: 100%; }
        .success-box { padding: 15px; background-color: #d4edda; color: #155724; border-radius: 10px; border: 1px solid #c3e6cb; margin-bottom: 10px; }
        .warning-box { padding: 15px; background-color: #fff3cd; color: #856404; border-radius: 10px; border: 1px solid #ffeeba; margin-bottom: 10px; }
        .error-box { padding: 15px; background-color: #f8d7da; color: #721c24; border-radius: 10px; border: 1px solid #f5c6cb; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Donchian Pro Scanner")

# --- 2. ASSET LISTS ---
COMMODITIES = {
    'Gold': 'GC=F', 'Silver': 'SI=F', 'Copper': 'HG=F',
    'Aluminum': 'ALI=F', 'Crude Oil': 'CL=F', 'Natural Gas': 'NG=F'
}

# FULL NIFTY 500 LIST
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
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@st.cache_data(ttl=900)
def fetch_and_scan(tickers, rsi_filter=False, trend_filter=False):
    try:
        data = yf.download(tickers, period="6mo", group_by='ticker', threads=True, progress=False)
    except:
        return [], []

    buys, exits = [], []

    for ticker in tickers:
        try:
            df = data[ticker] if len(tickers) > 1 else data
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            if len(df) < 50: continue

            # Strategy Indicators
            df['High_20'] = df['High'].rolling(20).max()
            df['Low_20'] = df['Low'].rolling(20).min()
            df['Middle'] = (df['High_20'] + df['Low_20']) / 2
            
            # Risk & Trend
            df['SMA_200'] = df['Close'].rolling(200).mean() if len(df) > 200 else df['Close'].rolling(50).mean()
            df['RSI'] = calculate_rsi(df['Close'])
            
            today, prev = df.iloc[-1], df.iloc[-2]
            name = ticker.replace('.NS', '')
            
            # Calculate Risk (Distance to Middle Band)
            risk_pct = round(((today['Close'] - today['Middle']) / today['Close']) * 100, 2)
            
            # --- BUY LOGIC ---
            if prev['Close'] < prev['Middle'] and today['Close'] > today['Middle']:
                if trend_filter and today['Close'] < today['SMA_200']: continue
                if rsi_filter and today['RSI'] > 70: continue
                
                buys.append({
                    "Stock": name,
                    "Price": round(today['Close'], 2),
                    "Stop Loss %": f"{risk_pct}%",
                    "RSI": round(today['RSI'], 1),
                    "Trend": "⬆️ UP" if today['Close'] > today['SMA_200'] else "⬇️ DOWN"
                })

            # --- EXIT LOGIC ---
            elif prev['Close'] > prev['Middle'] and today['Close'] < today['Middle']:
                exits.append({
                    "Stock": name,
                    "Price": round(today['Close'], 2),
                    "RSI": round(today['RSI'], 1)
                })

        except: continue
    
    return buys, exits

def analyze_single_stock(ticker):
    try:
        df = yf.download(ticker, period="1y", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        df['High_20'] = df['High'].rolling(20).max()
        df['Low_20'] = df['Low'].rolling(20).min()
        df['Middle'] = (df['High_20'] + df['Low_20']) / 2
        
        # Determine Status
        curr = df.iloc[-1]
        is_bullish = curr['Close'] > curr['Middle']
        
        # Find Signal Dates
        df['Buy_Signal'] = (df['Close'] > df['Middle']) & (df['Close'].shift(1) < df['Middle'].shift(1))
        df['Exit_Signal'] = (df['Close'] < df['Middle']) & (df['Close'].shift(1) > df['Middle'].shift(1))
        
        last_buy = df[df['Buy_Signal']].tail(1)
        
        result = {
            "name": ticker.replace('.NS', ''),
            "price": round(curr['Close'], 2),
            "is_bullish": is_bullish,
            "buy_date": "-", "buy_price": 0, "pnl": 0.0,
            "fresh_signal": False
        }
        
        if not last_buy.empty:
            buy_dt = last_buy.index[-1]
            result['buy_date'] = buy_dt.strftime('%d-%b-%Y')
            result['buy_price'] = round(last_buy['Close'].values[-1], 2)
            
            # Check if signal is FRESH (Today or Yesterday)
            today_dt = pd.Timestamp.now().normalize()
            if (today_dt - buy_dt).days <= 1:
                result['fresh_signal'] = True
                
            if is_bullish:
                result['pnl'] = round(((curr['Close'] - result['buy_price']) / result['buy_price']) * 100, 2)
                
        return result
    except: return None

# --- 4. APP UI ---
tab1, tab2 = st.tabs(["🚀 Scanner", "🕵️ Detective"])

with tab1:
    st.sidebar.header("🔍 Filters")
    market = st.sidebar.radio("Market", ["NSE Nifty 500", "Commodities"])
    use_trend = st.sidebar.checkbox("Trend Filter (200 SMA)", True)
    use_rsi = st.sidebar.checkbox("RSI Filter (< 70)", True)

    if st.button("RUN SCANNER", key="run_scan"):
        with st.spinner("Scanning markets..."):
            t_list = NSE_500_LIST if market == "NSE Nifty 500" else list(COMMODITIES.values())
            buys, exits = fetch_and_scan(t_list, use_rsi, use_trend)
            
            # Store results in session state for interactivity
            st.session_state['scan_results'] = buys
            
            c1, c2 = st.columns(2)
            with c1:
                st.success(f"✅ BUY SIGNALS ({len(buys)})")
                if buys: 
                    st.dataframe(pd.DataFrame(buys), hide_index=True, use_container_width=True)
                else: st.info("No Buy signals.")
            
            with c2:
                st.error(f"❌ EXIT SIGNALS ({len(exits)})")
                if exits: 
                    st.dataframe(pd.DataFrame(exits), hide_index=True, use_container_width=True)
                else: st.info("No Exit signals.")

    # --- DRILL DOWN FEATURE ---
    if 'scan_results' in st.session_state and st.session_state['scan_results']:
        st.markdown("---")
        st.subheader("🔍 Analyze a Stock from Results")
        
        # Create list of stock names from the scan
        stock_options = [x['Stock'] for x in st.session_state['scan_results']]
        selected = st.selectbox("Select stock to see details:", stock_options)
        
        if selected:
            # Add .NS if NSE
            search_t = selected + ".NS" if market == "NSE Nifty 500" else COMMODITIES.get(selected, selected)
            
            # Run Single Analysis
            data = analyze_single_stock(search_t)
            if data:
                 st.info(f"Analysis: {data['name']}")
                 c1, c2, c3 = st.columns(3)
                 c1.metric("Buy Date", data['buy_date'])
                 c2.metric("Entry Price", f"₹ {data['buy_price']}")
                 c3.metric("Current P&L", f"{data['pnl']}%", delta_color="normal")

with tab2:
    st.header("Search Any Stock")
    txt_input = st.text_input("Enter Symbol (e.g. RELIANCE)", "").upper().strip()
    
    if st.button("Check Status"):
        if txt_input:
            # Map Commodities
            t_search = txt_input
            rev_map = {k.split()[0].upper(): v for k, v in COMMODITIES.items()}
            if t_search in rev_map: t_search = rev_map[t_search]
            elif not t_search.endswith(".NS") and "=" not in t_search: t_search += ".NS"
            
            data = analyze_single_stock(t_search)
            
            if data:
                # --- SMART STATUS DISPLAY ---
                if data['is_bullish']:
                    if data['fresh_signal']:
                         st.markdown(f"""<div class="success-box"><span style="font-size:20px;">⚡ <b>FRESH BUY SIGNAL!</b></span><br>Signal generated in last 24hrs.<br>Price: ₹{data['price']}</div>""", unsafe_allow_html=True)
                    else:
                         st.markdown(f"""<div class="warning-box"><span style="font-size:20px;">📈 <b>EXISTING UPTREND</b></span><br>Stock is in buy zone, but signal is old ({data['buy_date']}).<br>Current P&L: <b>{data['pnl']}%</b></div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="error-box"><span style="font-size:20px;">❌ <b>NO SIGNAL / EXIT ZONE</b></span><br>Price is below Middle Band.<br>Current Price: ₹{data['price']}</div>""", unsafe_allow_html=True)
            else:
                st.error("Stock not found.")
