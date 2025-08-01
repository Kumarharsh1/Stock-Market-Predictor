import streamlit as st
import pandas as pd
import ta
import mplfinance as mpf
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="📈 Stock Signal Analyzer", layout="wide")
st.title("📈 Buy/Sell Signal Detection with 5-Condition Strategy")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # --- Clean columns ---
    df.columns = df.columns.str.strip().str.lower()
    if 'date' not in df.columns:
        st.error("Missing 'Date' column.")
        st.stop()

    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df.set_index('date', inplace=True)
    df = df.sort_index()

    # --- Ensure required columns ---
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            st.stop()

    # --- Indicator Calculation ---
    df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
    df['ema_fast'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema_slow'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()

    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_middle'] = bb.bollinger_mavg()

    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['volume_avg10'] = df['volume'].rolling(10).mean()

    # --- Signal Logic ---
    df['buy_signal'] = (
        (df['close'] > df['vwap']) &
        (df['ema_fast'] > df['ema_slow']) &
        (df['rsi'] > 40) & (df['rsi'] < 60) &
        (df['close'] <= df['bb_lower'] * 1.05) &
        (df['volume'] > df['volume_avg10'])
    )

    df['sell_signal'] = (
        (df['close'] < df['vwap']) &
        (df['ema_fast'] < df['ema_slow']) &
        ((df['rsi'] > 70) | (df['rsi'].diff() < -5)) &
        (df['close'] >= df['bb_upper'] * 0.95) &
        (df['volume'] > df['volume_avg10'])
    )

    # --- Markers for mplfinance ---
    df['buy_marker'] = np.where(df['buy_signal'], df['low'] * 0.98, np.nan)
    df['sell_marker'] = np.where(df['sell_signal'], df['high'] * 1.02, np.nan)

    # --- Show Buy/Sell message button ---
    last_signal = None
    if df['buy_signal'].iloc[-1]:
        last_signal = 'Buy'
    elif df['sell_signal'].iloc[-1]:
        last_signal = 'Sell'
    else:
        last_signal = 'Neutral'

    st.button(f"📢 Current Signal: {last_signal.upper()}")

    # --- Create addplots ---
    apds = [
        mpf.make_addplot(df['ema_fast'], color='blue'),
        mpf.make_addplot(df['ema_slow'], color='orange'),
        mpf.make_addplot(df['vwap'], color='purple', linestyle='--'),
        mpf.make_addplot(df['bb_upper'], color='gray', linestyle='--'),
        mpf.make_addplot(df['bb_lower'], color='gray', linestyle='--'),
        mpf.make_addplot(df['buy_marker'], type='scatter', marker='^', markersize=100, color='green'),
        mpf.make_addplot(df['sell_marker'], type='scatter', marker='v', markersize=100, color='red')
    ]

    # --- Plot chart ---
    st.subheader("Candlestick Chart with Strategy Signals")
    fig, _ = mpf.plot(
        df,
        type='candle',
        style='yahoo',
        volume=True,
        title="Buy/Sell Signal Chart",
        addplot=apds,
        figscale=1.6,
        figratio=(16, 9),
        returnfig=True
    )
    st.pyplot(fig)

else:
    st.info("Upload a stock CSV file with 'Date, Open, High, Low, Close, Volume' columns.")
