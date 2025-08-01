import streamlit as st
import pandas as pd
import ta
import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Stock Signal Analyzer", layout="wide")
st.title("📊 Stock Buy/Sell Signal Analyzer")

# --- Sidebar for controls ---
st.sidebar.header("Upload and Filters")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df.columns = df.columns.str.strip().str.lower()
    df.rename(columns={
        'date': 'date',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    }, inplace=True)

    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df.set_index('date', inplace=True)

    # --- Sidebar date range ---
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    start_date = st.sidebar.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

    df = df.loc[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
    df.dropna(inplace=True)

    # --- Sidebar indicator settings ---
    st.sidebar.header("Indicators")
    fast_ema_period = st.sidebar.slider("Fast EMA", 5, 30, 10)
    slow_ema_period = st.sidebar.slider("Slow EMA", 10, 60, 20)
    bb_period = st.sidebar.slider("Bollinger Bands Period", 10, 30, 20)
    bb_std = st.sidebar.slider("BB Std Dev", 1, 3, 2)
    rsi_period = st.sidebar.slider("RSI Period", 7, 21, 14)

    # --- Indicators ---
    df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
    df['ema_fast'] = ta.trend.ema_indicator(df['close'], fast_ema_period)
    df['ema_slow'] = ta.trend.ema_indicator(df['close'], slow_ema_period)
    bb = ta.volatility.BollingerBands(df['close'], bb_period, bb_std)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], rsi_period).rsi()
    df['vol_avg'] = df['volume'].rolling(10).mean()

    # --- Buy/Sell logic ---
    df['buy_signal'] = (
        (df['close'] > df['vwap']) &
        (df['ema_fast'] > df['ema_slow']) &
        (df['rsi'] > 40) & (df['rsi'] < 60) &
        (df['close'] <= df['bb_lower']) &
        (df['volume'] > df['vol_avg'])
    )

    df['sell_signal'] = (
        (df['close'] < df['vwap']) &
        (df['ema_fast'] < df['ema_slow']) &
        ((df['rsi'] > 70) | (df['rsi'].diff() < -10)) &
        (df['close'] >= df['bb_upper']) &
        (df['volume'] > df['vol_avg'])
    )

    df['buy_marker'] = np.where(df['buy_signal'], df['low'] * 0.98, np.nan)
    df['sell_marker'] = np.where(df['sell_signal'], df['high'] * 1.02, np.nan)

    # --- Latest Signal ---
    last_signal = "Neutral"
    last_color = "gray"
    last_buy = df[df['buy_signal']].tail(1)
    last_sell = df[df['sell_signal']].tail(1)

    if not last_buy.empty and (last_sell.empty or last_buy.index[-1] > last_sell.index[-1]):
        last_signal = "Buy"
        last_color = "green"
    elif not last_sell.empty:
        last_signal = "Sell"
        last_color = "red"

    st.markdown(f"### 📌 Latest Signal: <span style='color:{last_color}; font-size:20px'><b>{last_signal}</b></span>", unsafe_allow_html=True)

    # --- Candlestick Chart ---
    st.subheader("📈 Candlestick Chart with Buy/Sell Signals")
    apds = [
        mpf.make_addplot(df['ema_fast'], color='blue'),
        mpf.make_addplot(df['ema_slow'], color='orange'),
        mpf.make_addplot(df['vwap'], color='purple'),
        mpf.make_addplot(df['bb_upper'], color='gray', linestyle='--'),
        mpf.make_addplot(df['bb_lower'], color='gray', linestyle='--'),
        mpf.make_addplot(df['buy_marker'], type='scatter', marker='^', markersize=100, color='green'),
        mpf.make_addplot(df['sell_marker'], type='scatter', marker='v', markersize=100, color='red')
    ]

    fig, _ = mpf.plot(
        df,
        type='candle',
        volume=True,
        style='yahoo',
        addplot=apds,
        figscale=2.5,
        figratio=(18, 10),
        title="Zoomed Strategy Chart",
        returnfig=True
    )
    st.pyplot(fig)

    # --- Indicator Charts ---
    st.subheader("🔍 Indicators")
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))
    ax1.plot(df.index, df['rsi'], label='RSI', color='purple')
    ax1.axhline(70, color='red', linestyle='--')
    ax1.axhline(30, color='green', linestyle='--')
    ax1.legend()
    ax1.set_title("RSI")

    ax2.plot(df.index, df['volume'], label='Volume', color='gray')
    ax2.plot(df.index, df['vol_avg'], label='10-period Avg Volume', color='blue')
    ax2.set_title("Volume")
    ax2.legend()
    st.pyplot(fig2)

else:
    st.info("📤 Please upload a valid CSV file with columns: Date, Open, High, Low, Close, Volume")
