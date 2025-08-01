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

if 'sample_loaded' not in st.session_state:
    st.session_state.sample_loaded = False

if st.button("Load Sample Data (ADANIPORTS)"):
    try:
        sample_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/stocks.csv"
        sample_data = pd.read_csv(sample_url)
        sample_data.to_csv("sample_stock_data.csv", index=False)
        st.session_state.sample_loaded = True
        st.rerun()
    except Exception as e:
        st.error(f"Couldn't load sample data: {str(e)}")

st.sidebar.header("Settings")
st.sidebar.markdown("Configure the analysis parameters")

uploaded_file = st.sidebar.file_uploader(
    "Upload your stock data CSV file",
    type=["csv"],
    help="Upload a CSV file with columns: Date, Open, High, Low, Close, Volume"
)

if uploaded_file is None and st.session_state.sample_loaded:
    uploaded_file = "sample_stock_data.csv"

with st.sidebar.expander("Technical Indicators"):
    ema_short = st.slider("Short EMA Period", 5, 50, 20)
    ema_long = st.slider("Long EMA Period", 20, 200, 50)
    bb_period = st.slider("Bollinger Bands Period", 10, 50, 20)
    bb_std = st.slider("Bollinger Bands Std Dev", 1, 3, 2)
    rsi_period = st.slider("RSI Period", 5, 30, 14)

with st.sidebar.expander("Date Range"):
    default_end = datetime.now()
    default_start = default_end - timedelta(days=365)
    start_date = st.date_input("Start Date", default_start)
    end_date = st.date_input("End Date", default_end)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.title()

        date_col = None
        for col in df.columns:
            if 'date' in col.lower():
                date_col = col
                break

        if date_col is None:
            st.error("No date column found.")
            st.stop()

        df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)

        # Keep only necessary columns
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [col for col in df.columns if col in required_cols]

        for col in required_cols:
            if col not in df.columns:
                st.warning(f"Column '{col}' not found. Using fallback/default.")
                if col == 'Volume':
                    df[col] = 1000000
                else:
                    st.stop()

        df = df[required_cols]

        df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
        if df.empty:
            st.warning("No data available in selected date range.")
            st.stop()

        df.set_index('Date', inplace=True)
        df.dropna(inplace=True)

        # Drop existing indicator columns if present
        indicators_to_drop = [
            f'EMA{ema_short}', f'EMA{ema_long}', 'RSI', 'VWAP',
            'MACD', 'MACD_signal', 'MACD_diff',
            'BB_upper', 'BB_middle', 'BB_lower',
            'Buy', 'Sell', 'Buy_Marker', 'Sell_Marker'
        ]
        for col in indicators_to_drop:
            if col in df.columns:
                df.drop(columns=col, inplace=True)

        st.subheader("Technical Indicators")

        df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume']).volume_weighted_average_price()
        df[f'EMA{ema_short}'] = ta.trend.EMAIndicator(df['Close'], ema_short).ema_indicator()
        df[f'EMA{ema_long}'] = ta.trend.EMAIndicator(df['Close'], ema_long).ema_indicator()

        bb = ta.volatility.BollingerBands(df['Close'], bb_period, bb_std)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()

        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], rsi_period).rsi()

        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()

        df['Buy'] = (
            (df[f'EMA{ema_short}'] > df[f'EMA{ema_long}']) &
            (df['Close'] > df['VWAP']) &
            (df['RSI'] > 30) &
            (df['MACD'] > df['MACD_signal'])
        )

        df['Sell'] = (
            (df[f'EMA{ema_short}'] < df[f'EMA{ema_long}']) &
            (df['Close'] < df['VWAP']) &
            (df['RSI'] < 70) &
            (df['MACD'] < df['MACD_signal'])
        )

        df['Buy_Marker'] = np.where(df['Buy'], df['Low'] * 0.98, np.nan)
        df['Sell_Marker'] = np.where(df['Sell'], df['High'] * 1.02, np.nan)

        st.subheader("Trading Signals")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Recent Buy Signals**")
            buy_signals = df[df['Buy']].tail(5)
            st.dataframe(buy_signals[["Close", f"EMA{ema_short}", f"EMA{ema_long}", "VWAP", "RSI"]])

        with col2:
            st.write("**Recent Sell Signals**")
            sell_signals = df[df['Sell']].tail(5)
            st.dataframe(sell_signals[["Close", f"EMA{ema_short}", f"EMA{ema_long}", "VWAP", "RSI"]])

        st.subheader("Candlestick Chart with Signals")
        if df.empty:
    st.error("No data available for the selected date range and indicators. Please adjust your inputs or upload a different file.")
else:
    # continue with candlestick plotting here

        apds = [
            mpf.make_addplot(df[f'EMA{ema_short}'], color='blue', width=1.5),
            mpf.make_addplot(df[f'EMA{ema_long}'], color='orange', width=1.5),
            mpf.make_addplot(df['VWAP'], color='purple', linestyle=':'),
            mpf.make_addplot(df['BB_upper'], color='gray', linestyle='--'),
            mpf.make_addplot(df['BB_lower'], color='gray', linestyle='--'),
            mpf.make_addplot(df['Buy_Marker'], type='scatter', marker='^', markersize=100, color='green'),
            mpf.make_addplot(df['Sell_Marker'], type='scatter', marker='v', markersize=100, color='red')
        ]

        fig, _ = mpf.plot(df, type='candle', style='yahoo', volume=True,
                          title='Candlestick Chart with Signals', addplot=apds,
                          returnfig=True, figscale=1.5, figratio=(12, 6))
        st.pyplot(fig)

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
            st.metric("RSI Value", f"{df['RSI'].iloc[-1]:.2f}")
            if df['RSI'].iloc[-1] > 70:
                st.error("Overbought Territory")
            elif df['RSI'].iloc[-1] < 30:
                st.success("Oversold Territory")
            else:
                st.info("Neutral Territory")

        with col3:
            macd_diff = df['MACD'].iloc[-1] - df['MACD_signal'].iloc[-1]
            st.metric("MACD Difference", f"{macd_diff:.4f}")
            if macd_diff > 0:
                st.success("Bullish Momentum")
            else:
                st.warning("Bearish Momentum")

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
        st.error(f"An error occurred: {str(e)}")
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
