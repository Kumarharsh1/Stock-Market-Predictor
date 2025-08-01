import streamlit as st
import pandas as pd
import numpy as np
import ta
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="Advanced Stock Signal App", layout="wide")

st.title("📈 Advanced Stock Market Signal Detector")

# File Upload
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV File", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # --- Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # --- Parse date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df.dropna(subset=['date'], inplace=True)
        df.set_index('date', inplace=True)
    else:
        st.error("CSV must contain 'Date' column.")
        st.stop()

    # --- Filter necessary columns
    try:
        df = df[['open', 'high', 'low', 'close', 'volume']]
    except KeyError as e:
        st.error(f"Missing columns: {e}")
        st.stop()

    # --- Sidebar Controls
    st.sidebar.subheader("📆 Select Date Range")
    min_date, max_date = df.index.min().date(), df.index.max().date()
    start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
    df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]

    st.sidebar.subheader("🧮 Technical Indicator Settings")
    fast_ema = st.sidebar.slider("Fast EMA", 5, 50, 20)
    slow_ema = st.sidebar.slider("Slow EMA", 10, 200, 50)
    rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
    bb_period = st.sidebar.slider("Bollinger Period", 10, 50, 20)
    bb_std = st.sidebar.slider("BB Std Dev", 1, 3, 2)

    # --- Indicators
    df['ema_fast'] = ta.trend.EMAIndicator(df['close'], window=fast_ema).ema_indicator()
    df['ema_slow'] = ta.trend.EMAIndicator(df['close'], window=slow_ema).ema_indicator()
    df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
    rsi = ta.momentum.RSIIndicator(df['close'], rsi_period)
    df['rsi'] = rsi.rsi()
    bb = ta.volatility.BollingerBands(df['close'], window=bb_period, window_dev=bb_std)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['volume_avg10'] = df['volume'].rolling(10).mean()

    # --- Signal Logic
    df['buy'] = (
        (df['close'] > df['vwap']) &
        (df['ema_fast'] > df['ema_slow']) &
        (df['rsi'].between(40, 60)) &
        (df['close'] <= df['bb_lower'] * 1.03) &
        (df['volume'] > df['volume_avg10'])
    )

    df['sell'] = (
        (df['close'] < df['vwap']) &
        (df['ema_fast'] < df['ema_slow']) &
        ((df['rsi'] > 70) | (df['rsi'].diff() < -5)) &
        (df['close'] >= df['bb_upper'] * 0.98) &
        (df['volume'] > df['volume_avg10'])
    )

    # --- Marker Points
    df['buy_marker'] = np.where(df['buy'], df['low'] * 0.98, np.nan)
    df['sell_marker'] = np.where(df['sell'], df['high'] * 1.02, np.nan)

    # --- Signal Summary
    st.subheader("🧭 Latest Signal")
    last_signal = "Neutral"
    signal_color = "gray"
    if df['buy'].iloc[-1]:
        last_signal = "📈 Buy"
        signal_color = "green"
    elif df['sell'].iloc[-1]:
        last_signal = "📉 Sell"
        signal_color = "red"

    st.markdown(f"<h2 style='color:{signal_color};text-align:center'>{last_signal}</h2>", unsafe_allow_html=True)

    # --- Candlestick Chart with Zoom (Large Scale)
    st.subheader("📊 Zoomed Candlestick Chart with Signals")

    apds = [
        mpf.make_addplot(df['ema_fast'], color='blue'),
        mpf.make_addplot(df['ema_slow'], color='orange'),
        mpf.make_addplot(df['vwap'], color='purple', linestyle='--'),
        mpf.make_addplot(df['bb_upper'], color='gray', linestyle='--'),
        mpf.make_addplot(df['bb_lower'], color='gray', linestyle='--'),
        mpf.make_addplot(df['buy_marker'], type='scatter', marker='^', color='green', markersize=200),
        mpf.make_addplot(df['sell_marker'], type='scatter', marker='v', color='red', markersize=200)
    ]

    fig, _ = mpf.plot(
        df,
        type='candle',
        style='yahoo',
        volume=True,
        title='Zoomed Strategy Chart',
        figscale=2.5,
        figratio=(18, 9),
        addplot=apds,
        returnfig=True
    )
    st.pyplot(fig)

    # --- Optional Indicator Chart
    with st.expander("📉 RSI and Volume Charts"):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        ax1.plot(df.index, df['rsi'], label='RSI', color='purple')
        ax1.axhline(70, color='red', linestyle='--')
        ax1.axhline(30, color='green', linestyle='--')
        ax1.set_title("RSI Indicator")
        ax1.legend()

        ax2.plot(df.index, df['volume'], label='Volume', color='blue')
        ax2.plot(df.index, df['volume_avg10'], label='10-Period Avg Vol', color='orange')
        ax2.set_title("Volume")
        ax2.legend()

        st.pyplot(fig)

else:
    st.info("Please upload a stock data CSV file with columns: Date, Open, High, Low, Close, Volume.")

