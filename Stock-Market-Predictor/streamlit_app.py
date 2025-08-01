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

# --- State Management for Data ---
if 'data' not in st.session_state:
    st.session_state.data = None

if st.button("Load Sample Data (AAPL)"):
    try:
        # Using a suitable single-stock data source
        sample_url = "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv"
        df = pd.read_csv(sample_url)
        # Standardize column names to match expected format
        df.columns = df.columns.str.strip().str.title()
        df.rename(columns={'Date': 'Date', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'}, inplace=True)
        st.session_state.data = df
        st.info("Sample data loaded successfully!")
    except Exception as e:
        st.error(f"Couldn't load sample data: {str(e)}")
        st.session_state.data = None

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
        df_uploaded = pd.read_csv(uploaded_file)
        st.session_state.data = df_uploaded
    except Exception as e:
        st.error(f"Error reading the uploaded file: {str(e)}")
        st.session_state.data = None

# --- Main App Logic ---
if 'data' in st.session_state and st.session_state.data is not None and not st.session_state.data.empty:
    df = st.session_state.data.copy()

    # --- CRITICAL FIX 1: Handle multi-level or tuple-based column names ---
    new_columns = []
    for col in df.columns:
        if isinstance(col, tuple):
            new_columns.append("_".join(str(c) for c in col))
        else:
            new_columns.append(col)
    df.columns = new_columns
    
    # Rename all columns to lowercase for the ta library
    df.columns = [col.lower() for col in df.columns]
    
    # --- CRITICAL FIX 2: Explicitly handle date format to prevent warnings and errors ---
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df.dropna(subset=['date'], inplace=True)
        df.set_index('date', inplace=True)
    else:
        st.error("The CSV file must have a 'Date' column.")
        st.stop()
        
    st.subheader("Data Preview")
    st.write(df.tail())
    
    try:
        with st.sidebar.expander("Date Range"):
            if not df.empty:
                min_date = df.index.min().date() if pd.notna(df.index.min()) else datetime.now().date() - timedelta(days=365*2)
                max_date = df.index.max().date() if pd.notna(df.index.max()) else datetime.now().date()
            else:
                min_date = datetime.now().date() - timedelta(days=365*2)
                max_date = datetime.now().date()

            default_start = max_date - timedelta(days=365)
            
            start_date = st.date_input("Start Date", value=default_start, min_value=min_date, max_value=max_date)
            end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
        
        df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
        if df.empty:
            st.warning("No data available in selected date range.")
            st.stop()

        df.dropna(inplace=True)
        # The column names are already lowercase, so this line is no longer necessary:
        # df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        
        with st.sidebar.expander("Technical Indicators"):
            ema_short = st.slider("Short EMA Period", 5, 50, 20)
            ema_long = st.slider("Long EMA Period", 20, 200, 50)
            bb_period = st.slider("Bollinger Bands Period", 10, 50, 20)
            bb_std = st.slider("Bollinger Bands Std Dev", 1, 3, 2)
            rsi_period = st.slider("RSI Period", 5, 30, 14)

        # --- Technical Indicator Calculation ---
        st.subheader("Technical Indicators")
        df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
        df[f'ema{ema_short}'] = ta.trend.EMAIndicator(df['close'], ema_short).ema_indicator()
        df[f'ema{ema_long}'] = ta.trend.EMAIndicator(df['close'], ema_long).ema_indicator()

        bb = ta.volatility.BollingerBands(df['close'], bb_period, bb_std)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()

        df['rsi'] = ta.momentum.RSIIndicator(df['close'], rsi_period).rsi()

        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # --- Buy/Sell Signal Logic ---
        df['buy'] = ((df[f'ema{ema_short}'] > df[f'ema{ema_long}']) & (df['close'] > df['vwap']) & (df['rsi'] > 30) & (df['macd'] > df['macd_signal']))
        df['sell'] = ((df[f'ema{ema_short}'] < df[f'ema{ema_long}']) & (df['close'] < df['vwap']) & (df['rsi'] < 70) & (df['macd'] < df['macd_signal']))
        
        # Ensure signals are only plotted on the actual day of the signal
        df['buy_marker'] = np.where(df['buy'].diff() == 1, df['low'] * 0.98, np.nan)
        df['sell_marker'] = np.where(df['sell'].diff() == 1, df['high'] * 1.02, np.nan)

        st.subheader("Trading Signals")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Recent Buy Signals**")
            buy_signals = df[df['buy_marker'].notna()].tail(5)
            st.dataframe(buy_signals[["close", f"ema{ema_short}", f"ema{ema_long}", "vwap", "rsi"]])

        with col2:
            st.write("**Recent Sell Signals**")
            sell_signals = df[df['sell_marker'].notna()].tail(5)
            st.dataframe(sell_signals[["close", f"ema{ema_short}", f"ema{ema_long}", "vwap", "rsi"]])

        # --- Visualization - Candlestick Chart with Signals ---
        st.subheader("Candlestick Chart with Signals")
        apds = [
            mpf.make_addplot(df[f'ema{ema_short}'], color='blue', width=1.5, panel=0),
            mpf.make_addplot(df[f'ema{ema_long}'], color='orange', width=1.5, panel=0),
            mpf.make_addplot(df['vwap'], color='purple', linestyle=':', width=1.5, panel=0),
            mpf.make_addplot(df['bb_upper'], color='gray', linestyle='--', width=1, panel=0),
            mpf.make_addplot(df['bb_lower'], color='gray', linestyle='--', width=1, panel=0),
            mpf.make_addplot(df['buy_marker'], type='scatter', marker='^', markersize=100, color='green', panel=0),
            mpf.make_addplot(df['sell_marker'], type='scatter', marker='v', markersize=100, color='red', panel=0)
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
            if df['buy'].iloc[-1]:
                last_signal = "Bullish"
                last_color = "green"
            elif df['sell'].iloc[-1]:
                last_signal = "Bearish"
                last_color = "red"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Signal", last_signal)
            st.markdown(f"""<div style='background-color:{last_color};padding:10px;border-radius:5px;text-align:center;color:white'>
                            {last_signal} Conditions</div>""", unsafe_allow_html=True)
        with col2:
            if not df.empty:
                st.metric("RSI Value", f"{df['rsi'].iloc[-1]:.2f}")
                if df['rsi'].iloc[-1] > 70:
                    st.error("Overbought Territory")
                elif df['rsi'].iloc[-1] < 30:
                    st.success("Oversold Territory")
                else:
                    st.info("Neutral Territory")
            else:
                st.info("Not enough data to calculate RSI.")

        with col3:
            if not df.empty:
                macd_diff = df['macd'].iloc[-1] - df['macd_signal'].iloc[-1]
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

        ax1.plot(df.index, df['rsi'], label='RSI', color='purple')
        ax1.axhline(70, color='red', linestyle='--', alpha=0.5)
        ax1.axhline(30, color='green', linestyle='--', alpha=0.5)
        ax1.set_title('RSI')
        ax1.legend()

        ax2.plot(df.index, df['macd'], label='MACD', color='blue')
        ax2.plot(df.index, df['macd_signal'], label='Signal Line', color='orange')
        ax2.bar(df.index, df['macd_diff'], color=np.where(df['macd_diff'] > 0, 'green', 'red'))
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
