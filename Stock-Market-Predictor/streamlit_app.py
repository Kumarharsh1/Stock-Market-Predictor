import streamlit as st
import pandas as pd
import ta
import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="📈 Stock Predictor", layout="wide")
st.title("📈 Advanced Stock Market Buy/Sell Signal App")

# --- Upload Section ---
st.sidebar.header("Upload CSV File")
file = st.sidebar.file_uploader("Upload stock CSV with Date, Open, High, Low, Close, Volume", type=["csv"])

if file:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()

    if 'date' not in df.columns:
        st.error("❌ 'Date' column missing in uploaded CSV.")
        st.stop()

    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df.set_index('date', inplace=True)

    df = df[['open', 'high', 'low', 'close', 'volume']].dropna()

    # --- Sidebar: Indicator Configuration ---
    st.sidebar.header("Indicators")
    fast = st.sidebar.slider("Fast EMA", 5, 50, 20)
    slow = st.sidebar.slider("Slow EMA", 20, 200, 50)
    rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
    bb_period = st.sidebar.slider("Bollinger Period", 10, 40, 20)
    bb_std = st.sidebar.slider("BB Std Dev", 1, 3, 2)

    # --- Date Filter ---
    st.sidebar.header("Date Filter")
    start = st.sidebar.date_input("Start Date", df.index.min().date())
    end = st.sidebar.date_input("End Date", df.index.max().date())
    df = df[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]

    if df.empty:
        st.error("No data in selected range.")
        st.stop()

    # --- Indicators ---
    df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
    df['ema_fast'] = ta.trend.ema_indicator(df['close'], fast).fillna(method='bfill')
    df['ema_slow'] = ta.trend.ema_indicator(df['close'], slow).fillna(method='bfill')
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], rsi_period).rsi().fillna(method='bfill')
    bb = ta.volatility.BollingerBands(df['close'], window=bb_period, window_dev=bb_std)
    df['bb_upper'] = bb.bollinger_hband().fillna(method='bfill')
    df['bb_lower'] = bb.bollinger_lband().fillna(method='bfill')
    df['vol_avg10'] = df['volume'].rolling(10).mean()

    # --- Signal Logic ---
    df['buy'] = (
        (df['close'] > df['vwap']) &
        (df['ema_fast'] > df['ema_slow']) &
        (df['rsi'].between(40, 60)) &
        (df['close'] <= df['bb_lower'] * 1.05) &
        (df['volume'] > df['vol_avg10'])
    )

    df['sell'] = (
        (df['close'] < df['vwap']) &
        (df['ema_fast'] < df['ema_slow']) &
        ((df['rsi'] > 70) | (df['rsi'].diff() < -5)) &
        (df['close'] >= df['bb_upper'] * 0.95) &
        (df['volume'] > df['vol_avg10'])
    )

    df['buy_marker'] = np.where(df['buy'], df['low'] * 0.98, np.nan)
    df['sell_marker'] = np.where(df['sell'], df['high'] * 1.02, np.nan)

    st.subheader("Buy/Sell Signal Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Last Signal", "BUY" if df['buy'].iloc[-1] else ("SELL" if df['sell'].iloc[-1] else "NEUTRAL"))
    with col2:
        st.metric("RSI", f"{df['rsi'].iloc[-1]:.2f}")

    # --- Signal Bar ---
    signal_text = "BUY ✅" if df['buy'].iloc[-1] else ("SELL ❌" if df['sell'].iloc[-1] else "NEUTRAL ⚖️")
    signal_color = "green" if df['buy'].iloc[-1] else ("red" if df['sell'].iloc[-1] else "gray")
    st.markdown(f"""
        <div style='text-align:center;background-color:{signal_color};color:white;padding:10px;border-radius:8px;'>
            <h3>{signal_text}</h3>
        </div>
    """, unsafe_allow_html=True)

    # --- Chart ---
    apds = [
        mpf.make_addplot(df['ema_fast'], color='blue'),
        mpf.make_addplot(df['ema_slow'], color='orange'),
        mpf.make_addplot(df['vwap'], color='purple', linestyle='--'),
        mpf.make_addplot(df['bb_upper'], color='gray', linestyle='--'),
        mpf.make_addplot(df['bb_lower'], color='gray', linestyle='--'),
    ]

    if not df['buy_marker'].dropna().empty:
        apds.append(mpf.make_addplot(df['buy_marker'], type='scatter', color='green', marker='^', markersize=200))

    if not df['sell_marker'].dropna().empty:
        apds.append(mpf.make_addplot(df['sell_marker'], type='scatter', color='red', marker='v', markersize=200))

    fig, _ = mpf.plot(
        df,
        type='candle',
        volume=True,
        style='yahoo',
        addplot=apds,
        figscale=2.5,
        figratio=(18, 10),
        title="Strategy Chart with Buy/Sell Signals",
        returnfig=True
    )
    st.pyplot(fig)

else:
    st.info("Please upload a valid CSV file to begin.")
