import streamlit as st
import pandas as pd
import ta
import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="📈 Smart Trading Signals", layout="wide")
st.title("📈 Stock Buy/Sell Signal Predictor")

# Upload CSV File
st.sidebar.header("Upload Data")
file = st.sidebar.file_uploader("Upload a CSV with columns: Date, Open, High, Low, Close, Volume")

# Indicator settings
st.sidebar.header("Indicator Settings")
ema_fast_period = st.sidebar.slider("Fast EMA Period", 5, 50, 20)
ema_slow_period = st.sidebar.slider("Slow EMA Period", 20, 200, 50)
bb_period = st.sidebar.slider("Bollinger Bands Period", 10, 50, 20)
bb_std = st.sidebar.slider("Bollinger Bands Std Dev", 1, 3, 2)
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)

# Date Range selection
st.sidebar.header("Date Range")
def_date = datetime.today()
start_date = st.sidebar.date_input("Start Date", def_date - timedelta(days=180))
end_date = st.sidebar.date_input("End Date", def_date)

if file:
    df = pd.read_csv(file)
    df.columns = df.columns.str.lower().str.strip()

    if 'date' not in df.columns:
        st.error("CSV must contain a 'Date' column.")
        st.stop()

    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df.set_index('date', inplace=True)
    df = df.loc[start_date:end_date]
    df.dropna(inplace=True)

    # Calculate indicators
    df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
    df['ema_fast'] = ta.trend.EMAIndicator(df['close'], ema_fast_period).ema_indicator()
    df['ema_slow'] = ta.trend.EMAIndicator(df['close'], ema_slow_period).ema_indicator()
    bb = ta.volatility.BollingerBands(df['close'], bb_period, bb_std)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], rsi_period).rsi()
    df['vol_avg10'] = df['volume'].rolling(10).mean()

    # Buy and Sell Signal Logic
    df['buy_signal'] = (
        (df['close'] > df['vwap']) &
        (df['ema_fast'] > df['ema_slow']) &
        (df['rsi'] > 40) & (df['rsi'] < 60) &
        (df['close'] < df['bb_lower'] * 1.02) &
        (df['volume'] > df['vol_avg10'])
    )

    df['sell_signal'] = (
        (df['close'] < df['vwap']) &
        (df['ema_fast'] < df['ema_slow']) &
        (df['rsi'] > 70) &
        (df['close'] > df['bb_upper'] * 0.98) &
        (df['volume'] > df['vol_avg10'])
    )

    # Add markers
    df['buy_marker'] = np.where(df['buy_signal'], df['low'] * 0.98, np.nan)
    df['sell_marker'] = np.where(df['sell_signal'], df['high'] * 1.02, np.nan)

    # Show recent signals
    st.subheader("📊 Recent Trading Signals")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Buy Signals**")
        st.dataframe(df[df['buy_signal']][['close', 'rsi', 'vwap']].tail(5))
    with col2:
        st.write("**Sell Signals**")
        st.dataframe(df[df['sell_signal']][['close', 'rsi', 'vwap']].tail(5))

    # Signal Status Box
    latest_status = "Neutral"
    if df['buy_signal'].iloc[-1]:
        latest_status = "Buy Signal 📈"
    elif df['sell_signal'].iloc[-1]:
        latest_status = "Sell Signal 📉"

    st.markdown("---")
    st.subheader("📍 Market Decision Box")
    st.markdown(f"""
        <div style='padding:15px;background:#111;color:white;text-align:center;border-radius:10px;font-size:24px;'>
        <strong>{latest_status}</strong>
        </div>
    """, unsafe_allow_html=True)

    # Plot Chart
    st.subheader("🕯️ Candlestick Chart with Signals")
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
        style='yahoo',
        addplot=apds,
        volume=True,
        figscale=2.0,
        figratio=(16, 8),
        returnfig=True,
        title="Zoomed Strategy Candlestick Chart"
    )
    st.pyplot(fig)

else:
    st.info("Upload your CSV file to begin analysis.")
