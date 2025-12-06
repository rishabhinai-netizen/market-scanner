import streamlit as st
import yfinance as yf
import pandas as pd
import math

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Donchian 500 Scanner", layout="wide")
st.title("⚡ Donchian Channel Scanner (Nifty 500 + Commodities)")
st.markdown("""
<style>
    div.stButton > button { background-color: #00C853; color: white; font-weight: bold; border: none; }
    .stProgress > div > div > div > div { background-color: #00C853; }
</style>
""", unsafe_allow_html=True)

# --- 2. HARDCODED ASSET LISTS ---
# Commodities (International Futures as proxies for MCX)
COMMODITIES = {
    'Gold (Global)': 'GC=F', 'Silver (Global)': 'SI=F', 'Copper (Global)': 'HG=F',
    'Aluminum': 'ALI=F', 'Crude Oil': 'CL=F', 'Natural Gas': 'NG=F'
}

# Nifty 500 List (Hardcoded from your upload)
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

# --- 3. HELPER FUNCTIONS ---
def get_donchian_signal(df):
    """
    Returns 'BUY', 'EXIT' or 'NONE' based on Donchian Middle Band crossover.
    Strategy: 
    - BUY: Close crosses ABOVE Middle Band from below.
    - EXIT: Close crosses BELOW Middle Band from above.
    """
    if len(df) < 22: return None, 0
    
    # Calculate Middle Band (20 period)
    df['High_20'] = df['High'].rolling(20).max()
    df['Low_20'] = df['Low'].rolling(20).min()
    df['Middle'] = (df['High_20'] + df['Low_20']) / 2
    
    # Get last two days
    today = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Buy Condition
    if prev['Close'] < prev['Middle'] and today['Close'] > today['Middle']:
        return "BUY", today['Close']
    
    # Exit Condition
    elif prev['Close'] > prev['Middle'] and today['Close'] < today['Middle']:
        return "EXIT", today['Close']
        
    return "NONE", 0

def find_last_historical_signal(ticker):
    """Finds the last time a signal occurred in history."""
    try:
        df = yf.download(ticker, period="1y", progress=False)
        if df.empty: return None, None, None
        
        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df['High_20'] = df['High'].rolling(20).max()
        df['Low_20'] = df['Low'].rolling(20).min()
        df['Middle'] = (df['High_20'] + df['Low_20']) / 2
        
        df['Prev_Close'] = df['Close'].shift(1)
        df['Prev_Middle'] = df['Middle'].shift(1)
        
        df['Buy'] = (df['Prev_Close'] < df['Prev_Middle']) & (df['Close'] > df['Middle'])
        df['Exit'] = (df['Prev_Close'] > df['Prev_Middle']) & (df['Close'] < df['Middle'])
        
        last_buy = df[df['Buy']].tail(1)
        last_exit = df[df['Exit']].tail(1)
        
        buy_info = (last_buy.index[-1].date(), round(last_buy['Close'].values[-1], 2)) if not last_buy.empty else ("None", 0)
        exit_info = (last_exit.index[-1].date(), round(last_exit['Close'].values[-1], 2)) if not last_exit.empty else ("None", 0)
        
        current_status = "BULLISH (In Trade)" if df['Close'].iloc[-1] > df['Middle'].iloc[-1] else "BEARISH (Out of Trade)"
        
        return buy_info, exit_info, current_status
    except:
        return None, None, None

# --- 4. APP INTERFACE ---
tab1, tab2 = st.tabs(["🚀 Market Scanner", "🔍 Check Specific Stock"])

# === TAB 1: SCANNER ===
with tab1:
    st.header("Daily Scanner")
    market_choice = st.radio("Select Market:", ["NSE Nifty 500 (Hardcoded)", "Commodities"], horizontal=True)
    
    if st.button("RUN SCANNER", key="scan"):
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        if market_choice == "Commodities":
            scan_list = list(COMMODITIES.values())
            display_map = {v: k for k, v in COMMODITIES.items()}
        else:
            scan_list = NSE_500_LIST
            display_map = {}
            
        buy_signals = []
        exit_signals = []
        
        # We chunk requests to avoid timeouts with 500 stocks
        chunk_size = 50
        total_chunks = math.ceil(len(scan_list) / chunk_size)
        
        for i in range(total_chunks):
            chunk = scan_list[i*chunk_size : (i+1)*chunk_size]
            status_text.text(f"Scanning batch {i+1} of {total_chunks}...")
            
            try:
                # Bulk Download
                data = yf.download(chunk, period="3mo", group_by='ticker', threads=True, progress=False)
                
                for ticker in chunk:
                    try:
                        # Extract individual stock data
                        df = data[ticker] if len(chunk) > 1 else data
                        if df.empty: continue
                        
                        # Fix columns if needed
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                        
                        df = df.dropna()
                        signal, price = get_donchian_signal(df)
                        
                        name = display_map.get(ticker, ticker.replace('.NS', ''))
                        
                        if signal == "BUY":
                            buy_signals.append({"Stock": name, "Price": round(price, 2)})
                        elif signal == "EXIT":
                            exit_signals.append({"Stock": name, "Price": round(price, 2)})
                    except:
                        continue
            except:
                pass
            
            progress_bar.progress((i + 1) / total_chunks)
            
        status_text.text("Scan Complete!")
        progress_bar.empty()
        
        # Display Results
        c1, c2 = st.columns(2)
        with c1:
            st.success(f"✅ BUY SIGNALS ({len(buy_signals)})")
            if buy_signals:
                st.dataframe(pd.DataFrame(buy_signals), use_container_width=True, hide_index=True)
            else:
                st.write("No Buy signals today.")
                
        with c2:
            st.error(f"❌ EXIT SIGNALS ({len(exit_signals)})")
            if exit_signals:
                st.dataframe(pd.DataFrame(exit_signals), use_container_width=True, hide_index=True)
            else:
                st.write("No Exit signals today.")

# === TAB 2: CHECKER ===
with tab2:
    st.header("Stock Signal History")
    st.markdown("Check when a specific stock last triggered a signal.")
    
    user_input = st.text_input("Enter Symbol (e.g., RELIANCE, TATASTEEL, GOLD, SILVER)", "").upper().strip()
    
    if st.button("Check History"):
        if user_input:
            # Handle commodity names map
            ticker_search = user_input
            rev_map = {k.split()[0].upper(): v for k, v in COMMODITIES.items()} # Map GOLD -> GC=F
            if ticker_search in rev_map:
                ticker_search = rev_map[ticker_search]
            elif not ticker_search.endswith(".NS") and "=" not in ticker_search:
                ticker_search += ".NS"
                
            with st.spinner(f"Analyzing {ticker_search}..."):
                last_buy, last_exit, status = find_last_historical_signal(ticker_search)
                
                if status:
                    st.subheader(f"{user_input}: {status}")
                    m1, m2 = st.columns(2)
                    m1.metric("Last BUY Signal", f"{last_buy[0]}", f"₹ {last_buy[1]}")
                    m2.metric("Last EXIT Signal", f"{last_exit[0]}", f"₹ {last_exit[1]}")
                else:
                    st.error("Could not fetch data. Please check the spelling.")
