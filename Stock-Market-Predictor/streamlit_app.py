import streamlit as st
import pandas as pd
import ta
import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import os

st.set_page_config(
    page_title="Stock Market Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Stock Market Predictor")
st.markdown("""
Analyze stock data, detect Buy/Sell signals using technical indicators, and visualize market sentiment.
Upload your stock data CSV file to get started.
""")

# --- State Management for Sample Data ---
if 'sample_loaded' not in st.session_state:
    st.session_state.sample_loaded = False
if 'data' not in st.session_state:
    st.session_state.data = None

if st.button("Load Sample Data (AAPL)"):
    try:
        # Using a more suitable single-stock data source
        sample_url = "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv"
        df = pd.read_csv(sample_url)
        df.columns = df.columns.str.title()
        df.rename(columns={'Date': 'Date', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'}, inplace=True)
        
        # Save to session state instead of a file
        st.session_state.data = df
        st.session_state.sample_loaded = True
        st.info("Sample data loaded successfully!")
    except Exception as e:
        st.error(f"Couldn't load sample data: {str(e)}")

st.sidebar.header("Settings")
st.sidebar.markdown("Configure the analysis parameters")

uploaded_file = st.sidebar.file_uploader(
    "Upload your stock data CSV file",
    type=["csv"],
    help="Upload a CSV file with columns: Date, Open, High, Low, Close, Volume"
)

# --- Data Loading Logic ---
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.data = df
        st.session_state.sample_loaded = False # Reset if a new file is uploaded
    except Exception as e:
        st.error(f"Error reading the uploaded file: {str(e)}")
        st.session_state.data = None

if st.session_state.data is not None:
    df = st.session_state.data.copy()
    
    with st.sidebar.expander("Technical Indicators"):
        ema_short = st.slider("Short EMA Period", 5, 50, 20)
        ema_long = st.slider("Long EMA Period", 20, 200, 50)
        bb_period = st.slider("Bollinger Bands Period", 10, 50, 20)
        bb_std = st.slider("Bollinger Bands Std Dev", 1, 3, 2)
        rsi_period = st.slider("RSI Period", 5, 30, 14)

    try:
        df.columns = df.columns.str.strip().str.title()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df.set_index('Date', inplace=True)

        with st.sidebar.expander("Date Range"):
            min_date = df.index.min().date() if not df.empty and pd.notna(df.index.min()) else datetime.now().date() - timedelta(days=365*2)
            max_date = df.index.max().date() if not df.empty and pd.notna(df.index.max()) else datetime.now().date()
            
            default_start = max_date - timedelta(days=365)
            
            start_date = st.date_input("Start Date", value=default_start, min_value=min_date, max_value=max_date)
            end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

        df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
        if df.empty:
            st.warning("No data available in selected date range. Please adjust your date range or use a different file.")
            st.stop()

        df.dropna(inplace=True)
        df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        
        # --- Technical Indicator Calculation ---
        st.subheader("Technical Indicators")
        df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
        df[f'EMA{ema_short}'] = ta.trend.EMAIndicator(df['close'], ema_short).ema_indicator()
        df[f'EMA{ema_long}'] = ta.trend.EMAIndicator(df['close'], ema_long).ema_indicator()

        bb = ta.volatility.BollingerBands(df['close'], bb_period, bb_std)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()

        df['RSI'] = ta.momentum.RSIIndicator(df['close'], rsi_period).rsi()

        macd = ta.trend.MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()

        # --- Buy/Sell Signal Logic ---
        df['Buy'] = ((df[f'EMA{ema_short}'] > df[f'EMA{ema_long}']) & (df['close'] > df['VWAP']) & (df['RSI'] > 30) & (df['MACD'] > df['MACD_signal']))
        df['Sell'] = ((df[f'EMA{ema_short}'] < df[f'EMA{ema_long}']) & (df['close'] < df['VWAP']) & (df['RSI'] < 70) & (df['MACD'] < df['MACD_signal']))
        
        # Ensure signals are only plotted on the actual day of the signal
        df['Buy_Marker'] = np.where(df['Buy'].diff() == 1, df['low'] * 0.98, np.nan)
        df['Sell_Marker'] = np.where(df['Sell'].diff() == 1, df['high'] * 1.02, np.nan)

        st.subheader("Trading Signals")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Recent Buy Signals**")
            buy_signals = df[df['Buy_Marker'].notna()].tail(5)
            st.dataframe(buy_signals[["close", f"EMA{ema_short}", f"EMA{ema_long}", "VWAP", "RSI"]])

        with col2:
            st.write("**Recent Sell Signals**")
            sell_signals = df[df['Sell_Marker'].notna()].tail(5)
            st.dataframe(sell_signals[["close", f"EMA{ema_short}", f"EMA{ema_long}", "VWAP", "RSI"]])

        # --- Visualization - Candlestick Chart with Signals ---
        st.subheader("Candlestick Chart with Signals")
        apds = [
            mpf.make_addplot(df[f'EMA{ema_short}'], color='blue', width=1.5, panel=0),
            mpf.make_addplot(df[f'EMA{ema_long}'], color='orange', width=1.5, panel=0),
            mpf.make_addplot(df['VWAP'], color='purple', linestyle=':', width=1.5, panel=0),
            mpf.make_addplot(df['BB_upper'], color='gray', linestyle='--', width=1, panel=0),
            mpf.make_addplot(df['BB_lower'], color='gray', linestyle='--', width=1, panel=0),
            mpf.make_addplot(df['Buy_Marker'], type='scatter', marker='^', markersize=100, color='green', panel=0),
            mpf.make_addplot(df['Sell_Marker'], type='scatter', marker='v', markersize=100, color='red', panel=0)
        ]

        fig, axes = mpf.plot(
            df,
            type='candle',
            volume=True,
            style='yahoo',
            title="Stock Analysis with Technical Indicators",
            addplot=apds,
            figscale=1.5,
            figratio=(12, 6),
            returnfig=True
        )
        st.pyplot(fig)

        # --- Market Sentiment Analysis ---
        st.subheader("Market Sentiment Analysis")
        last_signal = "Neutral"
        last_color = "gray"
        if not df.empty:
            if df['Buy'].iloc[-1]:
                last_signal = "Bullish"
                last_color = "green"
            elif df['Sell'].iloc[-1]:
                last_signal = "Bearish"
                last_color = "red"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Signal", last_signal)
            st.markdown(f"""<div style='background-color:{last_color};padding:10px;border-radius:5px;text-align:center;color:white'>
                            {last_signal} Conditions</div>""", unsafe_allow_html=True)
        with col2:
            if not df.empty:
                st.metric("RSI Value", f"{df['RSI'].iloc[-1]:.2f}")
                if df['RSI'].iloc[-1] > 70:
                    st.error("Overbought Territory")
                elif df['RSI'].iloc[-1] < 30:
                    st.success("Oversold Territory")
                else:
                    st.info("Neutral Territory")
            else:
                st.info("Not enough data to calculate RSI.")

        with col3:
            if not df.empty:
                macd_diff = df['MACD'].iloc[-1] - df['MACD_signal'].iloc[-1]
                st.metric("MACD Difference", f"{macd_diff:.4f}")
                if macd_diff > 0:
                    st.success("Bullish Momentum")
                else:
                    st.warning("Bearish Momentum")
            else:
                st.info("Not enough data to calculate MACD.")

        # --- Indicator Charts ---
        st.subheader("Indicator Charts")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(df.index, df['RSI'], label='RSI', color='purple')
        ax1.axhline(70, color='red', linestyle='--', alpha=0.5)
        ax1.axhline(30, color='green', linestyle='--', alpha=0.5)
        ax1.set_title('RSI')
        ax1.legend()

        ax2.plot(df.index, df['MACD'], label='MACD', color='blue')
        ax2.plot(df.index, df['MACD_signal'], label='Signal Line', color='orange')
        ax2.bar(df.index, df['MACD_diff'], color=np.where(df['MACD_diff'] > 0, 'green', 'red'))
        ax2.set_title('MACD')
        ax2.legend()

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        st.stop()
else:
    st.info("Please upload a CSV file or click 'Load Sample Data' to begin analysis.")
    st.markdown("""
    ### Expected CSV Format:
    ```csv
    Date,Open,High,Low,Close,Volume
    2023-01-01,150.2,152.5,149.8,151.3,2500000
    2023-01-02,151.5,153.1,150.9,152.4,1800000
    ```
    """)
