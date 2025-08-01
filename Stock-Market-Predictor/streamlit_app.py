import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import ta
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(
    page_title="📈 Stock Signal Analyzer",
    layout="wide"
)

st.title("📊 Stock Market Signal Analyzer")
st.markdown("Upload your CSV and get buy/sell signals based on 5-condition strategy.")

# Sidebar Upload and Controls
st.sidebar.header("📁 Upload Data & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df.dropna(subset=['date'], inplace=True)
        df.set_index('date', inplace=True)
    else:
        st.error("CSV must contain a 'Date' column.")
        st.stop()

    df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)

    # Sidebar date selection
    min_date, max_date = df.index.min(), df.index.max()
    start = st.sidebar.date_input("Start Date", min_value=min_date.date(), max_value=max_date.date(), value=(max_date - timedelta(days=60)).date())
    end = st.sidebar.date_input("End Date", min_value=min_date.date(), max_value=max_date.date(), value=max_date.date())

    df = df.loc[(df.index.date >= start) & (df.index.date <= end)]

    if df.empty:
        st.warning("No data in this date range. Try changing it.")
        st.stop()

    # Sidebar indicator selection
    st.sidebar.header("📐 Indicators")
    fast_period = st.sidebar.slider("Fast EMA", 5, 20, 10)
    slow_period = st.sidebar.slider("Slow EMA", 10, 100, 20)
    rsi_period = st.sidebar.slider("RSI Period", 5, 21, 14)
    bb_period = st.sidebar.slider("Bollinger Band Period", 10, 30, 20)
    volume_period = 10

    # Compute indicators
    df[f'ema{fast_period}'] = ta.trend.ema_indicator(df['close'], fast_period)
    df[f'ema{slow_period}'] = ta.trend.ema_indicator(df['close'], slow_period)
    df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
    df['rsi'] = ta.momentum.rsi(df['close'], rsi_period)
    bb = ta.volatility.BollingerBands(df['close'], window=bb_period)
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_upper'] = bb.bollinger_hband()
    df['vol_avg'] = df['volume'].rolling(volume_period).mean()

    # Buy/Sell Conditions
    df['buy_signal'] = (
        (df['close'] > df['vwap']) &
        (df[f'ema{fast_period}'] > df[f'ema{slow_period}']) &
        (df['rsi'] > 40) & (df['rsi'] < 60) &
        (df['close'] < df['bb_lower'] * 1.05) &
        (df['volume'] > df['vol_avg'])
    )

    df['sell_signal'] = (
        (df['close'] < df['vwap']) &
        (df[f'ema{fast_period}'] < df[f'ema{slow_period}']) &
        ((df['rsi'] > 70) | (df['rsi'].diff() < -5)) &
        (df['close'] > df['bb_upper'] * 0.95) &
        (df['volume'] > df['vol_avg'])
    )
    st.write("Matching Buy Conditions", df[
    (df['close'] > df['vwap']) &
    (df['ema_fast'] > df['ema_slow']) &
    (df['rsi'] > 40) & (df['rsi'] < 60) &
    (df['close'] < df['bb_lower'] * 1.02) &
    (df['volume'] > df['volume'].rolling(10).mean())
])
if df['buy_signal'].any():
    latest_buy = df[df['buy_signal']].index[-1]
    # then display that in the top bar even if it's out of current range


    df['buy_marker'] = np.where(df['buy_signal'], df['low'] * 0.98, np.nan)
    df['sell_marker'] = np.where(df['sell_signal'], df['high'] * 1.02, np.nan)

    # Final check before plotting
    if df.empty or df[['open', 'high', 'low', 'close']].isnull().any().any():
        st.warning("Insufficient data to render chart.")
        st.stop()

    # Candlestick Plot
    st.subheader("📈 Zoomed Candlestick Chart with Buy/Sell Signals")
    apds = [
        mpf.make_addplot(df[f'ema{fast_period}'], color='blue'),
        mpf.make_addplot(df[f'ema{slow_period}'], color='orange'),
        mpf.make_addplot(df['vwap'], color='purple', linestyle='--'),
        mpf.make_addplot(df['bb_lower'], color='gray', linestyle='--'),
        mpf.make_addplot(df['bb_upper'], color='gray', linestyle='--'),
        mpf.make_addplot(df['buy_marker'], type='scatter', color='green', markersize=150, marker='^'),
        mpf.make_addplot(df['sell_marker'], type='scatter', color='red', markersize=150, marker='v')
    ]

    fig, _ = mpf.plot(
        df,
        type='candle',
        volume=True,
        style='yahoo',
        addplot=apds,
        figscale=2.0,
        figratio=(18,10),
        title="Zoomed Strategy Candlestick Chart",
        returnfig=True
    )
    st.pyplot(fig)

    # Signal Bar at Bottom
    latest = df.iloc[-1]
    if latest['buy_signal']:
        st.success("📢 BUY Signal Detected")
    elif latest['sell_signal']:
        st.error("📢 SELL Signal Detected")
    else:
        st.info("📢 No Strong Signal - Neutral Zone")
else:
    st.info("Please upload a CSV file to begin analysis.")
