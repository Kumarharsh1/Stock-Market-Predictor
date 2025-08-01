import streamlit as st
import pandas as pd
import numpy as np
import ta
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="📈 Strategy Chart", layout="wide")
st.title("📊 Stock Strategy with Buy/Sell Signal")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['date'])
        df.set_index('date', inplace=True)
    else:
        st.error("No 'Date' column found!")
        st.stop()

    # Date Filter
    with st.sidebar.expander("📅 Date Range"):
        min_date, max_date = df.index.min(), df.index.max()
        start = st.date_input("Start", min_value=min_date, max_value=max_date, value=min_date)
        end = st.date_input("End", min_value=min_date, max_value=max_date, value=max_date)
        df = df.loc[start:end]

    # Indicators
    with st.sidebar.expander("🧮 Indicators"):
        ema_fast = st.slider("Fast EMA", 5, 50, 20)
        ema_slow = st.slider("Slow EMA", 20, 200, 50)
        rsi_period = st.slider("RSI Period", 5, 30, 14)
        bb_period = st.slider("Bollinger Period", 10, 30, 20)
        bb_std = st.slider("BB StdDev", 1, 3, 2)

    df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
    df['ema_fast'] = ta.trend.EMAIndicator(df['close'], ema_fast).ema_indicator()
    df['ema_slow'] = ta.trend.EMAIndicator(df['close'], ema_slow).ema_indicator()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], rsi_period).rsi()
    bb = ta.volatility.BollingerBands(df['close'], bb_period, bb_std)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['vol_avg10'] = df['volume'].rolling(10).mean()

    # Signal Logic
    df['buy'] = (
        (df['close'] > df['vwap']) &
        (df['ema_fast'] > df['ema_slow']) &
        (df['rsi'].between(40, 60)) &
        (df['close'] <= df['bb_lower'] * 1.03) &
        (df['volume'] > df['vol_avg10'])
    )

    df['sell'] = (
        (df['close'] < df['vwap']) &
        (df['ema_fast'] < df['ema_slow']) &
        ((df['rsi'] > 70) | (df['rsi'].diff() < -5)) &
        (df['close'] >= df['bb_upper'] * 0.97) &
        (df['volume'] > df['vol_avg10'])
    )

    df['buy_marker'] = np.where(df['buy'], df['low'] * 0.98, np.nan)
    df['sell_marker'] = np.where(df['sell'], df['high'] * 1.02, np.nan)

    st.subheader("📉 Candlestick Chart with Signals")
    apds = [
        mpf.make_addplot(df['ema_fast'], color='blue'),
        mpf.make_addplot(df['ema_slow'], color='orange'),
        mpf.make_addplot(df['vwap'], color='purple', linestyle=':'),
        mpf.make_addplot(df['bb_upper'], color='gray', linestyle='--'),
        mpf.make_addplot(df['bb_lower'], color='gray', linestyle='--'),
        mpf.make_addplot(df['buy_marker'], type='scatter', marker='^', color='green', markersize=150),
        mpf.make_addplot(df['sell_marker'], type='scatter', marker='v', color='red', markersize=150)
    ]

    fig, axes = mpf.plot(
        df,
        type='candle',
        volume=True,
        style='yahoo',
        title="Strategy Chart (Zoomed Signals)",
        addplot=apds,
        figscale=2.2,
        figratio=(18, 9),
        returnfig=True
    )
    st.pyplot(fig)

    # Signal Bar at Top/Bottom
    latest = df.iloc[-1]
    signal = "Neutral"
    color = "gray"
    if latest['buy']:
        signal = "BUY"
        color = "green"
    elif latest['sell']:
        signal = "SELL"
        color = "red"

    st.markdown(f"""
        <div style='background-color:{color};padding:15px;text-align:center;color:white;border-radius:8px;font-size:24px'>
        ✅ Current Signal: {signal}
        </div>
    """, unsafe_allow_html=True)

    # Show latest signals
    st.subheader("📍 Latest Buy/Sell Points")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Recent Buy Signals**")
        st.dataframe(df[df['buy']][['close', 'rsi', 'vwap']].tail(5))
    with col2:
        st.markdown("**Recent Sell Signals**")
        st.dataframe(df[df['sell']][['close', 'rsi', 'vwap']].tail(5))
