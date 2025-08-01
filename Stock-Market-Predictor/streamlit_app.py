import streamlit as st
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta

st.set_page_config(page_title="Zoomed Strategy Chart", layout="wide")

st.title("📈 Buy/Sell Signal Strategy Chart (Zoomable)")

# --- Upload CSV ---
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()

    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain: {', '.join(required_cols)}")
        st.stop()

    # Parse date & clean
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)

    # --- Indicator Calculation ---
    df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)

    df['vwap'] = ta.volume.VolumeWeightedAveragePrice(
        high=df['high'], low=df['low'], close=df['close'], volume=df['volume']
    ).volume_weighted_average_price()

    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()

    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    # --- Buy/Sell Logic ---
    df['buy_marker'] = np.where((df['close'] > df['ema20']) & (df['close'] > df['ema50']), df['low'] * 0.98, np.nan)
    df['sell_marker'] = np.where((df['close'] < df['ema20']) & (df['close'] < df['ema50']), df['high'] * 1.02, np.nan)

    st.sidebar.subheader("📅 Select Date Range")

    min_date = df.index.min().date()
    max_date = df.index.max().date()

    start_date = st.sidebar.date_input("Start", value=max_date - timedelta(days=60), min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End", value=max_date, min_value=min_date, max_value=max_date)

    df_range = df[(df.index.date >= start_date) & (df.index.date <= end_date)]
    if df_range.empty:
        st.warning("No data in selected range.")
        st.stop()

    # --- Left Panel: Indicator Values ---
    latest = df_range.iloc[-1]
    st.sidebar.subheader("📊 Latest Indicator Values")
    st.sidebar.metric("EMA20", f"{latest['ema20']:.2f}")
    st.sidebar.metric("EMA50", f"{latest['ema50']:.2f}")
    st.sidebar.metric("VWAP", f"{latest['vwap']:.2f}")
    st.sidebar.metric("RSI", f"{latest['rsi']:.2f}")
    st.sidebar.metric("MACD", f"{latest['macd']:.2f}")
    st.sidebar.metric("MACD Signal", f"{latest['macd_signal']:.2f}")

    # --- Chart ---
    st.subheader("📌 Zoomed Chart with Buy/Sell Signals")

    apds = [
        mpf.make_addplot(df_range['ema20'], color='blue', width=1.2),
        mpf.make_addplot(df_range['ema50'], color='orange', width=1.2),
        mpf.make_addplot(df_range['vwap'], color='purple', linestyle='--'),
        mpf.make_addplot(df_range['bb_upper'], color='gray', linestyle='--'),
        mpf.make_addplot(df_range['bb_lower'], color='gray', linestyle='--'),
        mpf.make_addplot(df_range['buy_marker'], type='scatter', marker='^', color='green', markersize=150),
        mpf.make_addplot(df_range['sell_marker'], type='scatter', marker='v', color='red', markersize=150)
    ]

    fig, _ = mpf.plot(
        df_range,
        type='candle',
        volume=True,
        style='yahoo',
        title="Zoomed Buy/Sell Strategy Chart",
        addplot=apds,
        figscale=2.5,
        figratio=(16, 9),
        returnfig=True,
        savefig='figure1_strategy_chart.png'
    )

    st.pyplot(fig)
    st.success("Chart saved as: `figure1_strategy_chart.png`")

    # Optional: show Buy/Sell table
    st.subheader("📌 Recent Buy/Sell Signals")
    col1, col2 = st.columns(2)
    with col1:
        st.write("✅ Buy Signals")
        st.dataframe(df_range[df_range['buy_marker'].notna()][['close', 'ema20', 'ema50']].tail(5))
    with col2:
        st.write("❌ Sell Signals")
        st.dataframe(df_range[df_range['sell_marker'].notna()][['close', 'ema20', 'ema50']].tail(5))

else:
    st.info("Upload a CSV file with columns: Date, Open, High, Low, Close, Volume.")
