import streamlit as st
import yfinance as yf
import pandas as pd
import math

# --- 1. CONFIGURATION & PRIVACY ---
st.set_page_config(page_title="Market Sniper", layout="wide", initial_sidebar_state="expanded")

# Hides Streamlit branding and "Manage App" buttons for viewers
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    div.stButton > button { background-color: #00C853; color: white; font-weight: bold; border: none; width: 100%; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Pro Trend Scanner")
st.markdown("Strategy: **Donchian Breakout (20)** with Volume & Trend Confirmation.")

# --- 2. ASSET LISTS (FULL NIFTY 500) ---
COMMODITIES = {
    'Gold (Global)': 'GC=F', 'Silver (Global)': 'SI=F', 'Copper (Global)': 'HG=F',
    'Aluminum': 'ALI=F', 'Crude Oil': 'CL=F', 'Natural Gas': 'NG=F'
}

# Full Nifty 500 List
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

# --- 3. CORE STRATEGY ---
@st.cache_data(ttl=3600)  # Cache data for 1 hour so friends don't break limits
def fetch_and_scan(tickers, use_trend_filter=False):
    try:
        data = yf.download(tickers, period="1y", group_by='ticker', threads=True, progress=False)
    except:
        return [], []

    buy_signals = []
    exit_signals = []

    for ticker in tickers:
        try:
            df = data[ticker] if len(tickers) > 1 else data
            # Handle formatting issues
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            
            if len(df) < 200: continue # Skip new listings if not enough data
            
            # 1. Indicators
            df['High_20'] = df['High'].rolling(20).max()
            df['Low_20'] = df['Low'].rolling(20).min()
            df['Middle'] = (df['High_20'] + df['Low_20']) / 2
            df['SMA_200'] = df['Close'].rolling(200).mean()
            
            # Volume Calc (Avg 20 days)
            df['Vol_Avg'] = df['Volume'].rolling(20).mean()

            # 2. Signal Logic
            today = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Name Cleaning
            name = ticker.replace('.NS', '')
            
            # Logic: BUY
            if prev['Close'] < prev['Middle'] and today['Close'] > today['Middle']:
                # Trend Filter Check
                if use_trend_filter and today['Close'] < today['SMA_200']:
                    continue # Skip if in downtrend
                
                # Volume Check
                vol_status = "🔥 HIGH" if today['Volume'] > (today['Vol_Avg'] * 1.5) else "Normal"
                
                buy_signals.append({
                    "Stock": name, 
                    "CMP (₹)": round(today['Close'], 2),
                    "Volume": vol_status,
                    "Trend (200SMA)": "Uptrend" if today['Close'] > today['SMA_200'] else "Downtrend"
                })
            
            # Logic: EXIT
            elif prev['Close'] > prev['Middle'] and today['Close'] < today['Middle']:
                exit_signals.append({
                    "Stock": name, 
                    "CMP (₹)": round(today['Close'], 2),
                    "Trend (200SMA)": "Uptrend" if today['Close'] > today['SMA_200'] else "Downtrend"
                })
                
        except:
            continue
            
    return buy_signals, exit_signals

# --- 4. APP INTERFACE ---
tab1, tab2 = st.tabs(["🚀 Market Scanner", "🔍 Check Single Stock"])

# === TAB 1: BULK SCANNER ===
with tab1:
    st.sidebar.header("Scanner Settings")
    market = st.sidebar.radio("Select Market", ["NSE Nifty 500", "Commodities"])
    
    # New Feature: Trend Filter
    trend_filter = False
    if market == "NSE Nifty 500":
        trend_filter = st.sidebar.checkbox("Only Buy in Uptrend (Above 200 SMA)?", value=False)
        st.sidebar.caption("Filters out weak stocks preventing false buy signals.")
    
    if st.button("RUN SCANNER"):
        with st.spinner("Fetching data from Yahoo Finance... (This takes 10-15 seconds)"):
            target_list = NSE_500_LIST if market == "NSE Nifty 500" else list(COMMODITIES.values())
            buys, exits = fetch_and_scan(target_list, trend_filter)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"✅ BUY SIGNALS ({len(buys)})")
                if buys:
                    df_buy = pd.DataFrame(buys)
                    st.dataframe(df_buy, use_container_width=True, hide_index=True)
                else:
                    st.info("No Buy signals found.")

            with col2:
                st.error(f"❌ EXIT SIGNALS ({len(exits)})")
                if exits:
                    df_exit = pd.DataFrame(exits)
                    st.dataframe(df_exit, use_container_width=True, hide_index=True)
                else:
                    st.info("No Exit signals found.")

# === TAB 2: SINGLE STOCK CHECK ===
with tab2:
    st.header("Check Specific Stock")
    symbol = st.text_input("Enter Symbol (e.g. TATASTEEL)", "").upper()
    
    if st.button("Check Now"):
        if symbol:
            # Map commodity names if needed
            t_search = symbol
            rev_map = {k.split()[0].upper(): v for k, v in COMMODITIES.items()}
            if t_search in rev_map: t_search = rev_map[t_search]
            elif not t_search.endswith(".NS") and "=" not in t_search: t_search += ".NS"
            
            try:
                df = yf.download(t_search, period="1y", progress=False)
                # Cleaning
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                
                df['High_20'] = df['High'].rolling(20).max()
                df['Low_20'] = df['Low'].rolling(20).min()
                df['Middle'] = (df['High_20'] + df['Low_20']) / 2
                
                curr = df.iloc[-1]
                status = "✅ BUY ZONE" if curr['Close'] > curr['Middle'] else "❌ EXIT ZONE"
                
                st.metric(label=f"Current Status of {symbol}", value=status, delta=f"CMP: {round(curr['Close'], 2)}")
                st.line_chart(df[['Close', 'Middle']].tail(100))
                
            except:
                st.error("Stock not found.")
