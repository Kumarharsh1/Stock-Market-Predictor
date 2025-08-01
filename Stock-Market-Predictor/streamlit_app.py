import streamlit as st
import pandas as pd
import numpy as np
import ta
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="📈 Stock Market Predictor", layout="wide")
st.title("📈 Stock Market Predictor")
st.markdown("""
Upload your stock CSV file with columns: Date, Open, High, Low, Close, Volume.
Indicators: VWAP, EMA, Bollinger Bands, RSI.
Signals: Buy/Sell with visual markers and market sentiment.
""")

# Sidebar Inputs
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
ema_fast_period = st.sidebar.slider("Fast EMA", 5, 30, 10)
ema_slow_period = st.sidebar.slider("Slow EMA", 20, 100, 50)
bb_period = st.sidebar.slider("Bollinger Band Period", 10, 50, 20)
bb_std = st.sidebar.slider("BB Std Dev", 1, 3, 2)
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()

    if 'date' not in df.columns or 'close' not in df.columns:
        st.error("CSV must have 'Date', 'Open', 'High', 'Low', 'Close', 'Volume' columns.")
        st.stop()

    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    st.sidebar.subheader("Date Range")
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    start = st.sidebar.date_input("Start Date", min_value=min_date, max_value=max_date, value=max_date - timedelta(days=60))
    end = st.sidebar.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)
    df = df.loc[start:end]

    df.dropna(inplace=True)
    if df.empty:
        st.warning("No data available for selected range.")
        st.stop()

    # Technical Indicators
    df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
    df['ema_fast'] = ta.trend.EMAIndicator(df['close'], ema_fast_period).ema_indicator()
    df['ema_slow'] = ta.trend.EMAIndicator(df['close'], ema_slow_period).ema_indicator()
    bb = ta.volatility.BollingerBands(df['close'], bb_period, bb_std)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], rsi_period).rsi()
    df['vol_ma10'] = df['volume'].rolling(10).mean()

    # Buy/Sell Signal Logic
    df['buy_signal'] = (
        (df['close'] > df['vwap']) &
        (df['ema_fast'] > df['ema_slow']) &
        (df['rsi'] > 40) & (df['rsi'] < 60) &
        (df['close'] < df['bb_lower'] * 1.02) &
        (df['volume'] > df['vol_ma10'])
    )

    df['sell_signal'] = (
        (df['close'] < df['vwap']) &
        (df['ema_fast'] < df['ema_slow']) &
        ((df['rsi'] > 70) | (df['rsi'].diff() < -5)) &
        (df['close'] > df['bb_upper'] * 0.98) &
        (df['volume'] > df['vol_ma10'])
    )

    df['buy_marker'] = np.where(df['buy_signal'], df['low'] * 0.98, np.nan)
    df['sell_marker'] = np.where(df['sell_signal'], df['high'] * 1.02, np.nan)

    # Chart
    st.subheader("Candlestick Chart with Signals")
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
        volume=True,
        style='yahoo',
        addplot=apds,
        figratio=(18, 10),
        figscale=2.0,
        returnfig=True
    )
    st.pyplot(fig)

    # Signal Message
    st.subheader("Current Signal")
    if not df.empty:
        last_row = df.iloc[-1]
        if last_row['buy_signal']:
            st.success("🔼 It's time to BUY!")
        elif last_row['sell_signal']:
            st.error("🔽 It's time to SELL!")
        else:
            st.info("⚪ Neutral signal")
    else:
        st.warning("No data to determine signal.")

    # Indicator Chart
    st.subheader("Indicators")
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    ax1.plot(df.index, df['rsi'], label='RSI', color='blue')
    ax1.axhline(70, linestyle='--', color='red')
    ax1.axhline(30, linestyle='--', color='green')
    ax1.set_title('RSI')
    ax1.legend()

    ax2.plot(df.index, df['volume'], label='Volume', color='black')
    ax2.plot(df.index, df['vol_ma10'], label='10-period MA', color='orange')
    ax2.set_title('Volume with MA')
    ax2.legend()

    st.pyplot(fig2)

else:
    st.info("Please upload a CSV file to begin analysis.")
