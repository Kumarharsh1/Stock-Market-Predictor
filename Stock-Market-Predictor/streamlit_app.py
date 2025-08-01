import streamlit as st
import pandas as pd
import numpy as np
import ta
import mplfinance as mpf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="📈 Smart Stock Predictor")

st.title("📈 Smart Stock Predictor")
st.markdown("Upload your CSV or load sample data to analyze trading signals using technical indicators.")

# --- Sidebar ---
st.sidebar.header("1️⃣ Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if st.sidebar.button("📥 Load Sample Data"):
    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv")
    df.columns = df.columns.str.strip().str.title()
    df = df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
else:
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a file or load sample data.")
        st.stop()

# --- Preprocessing ---
df.columns = [col.strip().lower() for col in df.columns]
if 'date' not in df.columns:
    st.error("CSV must contain a 'Date' column.")
    st.stop()
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
df.dropna(subset=['date'], inplace=True)
df.set_index('date', inplace=True)
df = df[['open', 'high', 'low', 'close', 'volume']].dropna()

# --- Sidebar Controls ---
st.sidebar.header("2️⃣ Date Range")
min_date = df.index.min().date()
max_date = df.index.max().date()
start_date = st.sidebar.date_input("Start Date", min_value=min_date, max_value=max_date, value=max_date - timedelta(days=90))
end_date = st.sidebar.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

df = df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)]
if df.empty:
    st.error("No data in selected range.")
    st.stop()

# --- Indicator Inputs ---
st.sidebar.header("3️⃣ Technical Indicators")
fast_ema = st.sidebar.slider("Fast EMA", 5, 50, 20)
slow_ema = st.sidebar.slider("Slow EMA", 10, 200, 50)
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
bb_period = st.sidebar.slider("Bollinger Band Period", 10, 50, 20)
bb_std = st.sidebar.slider("BB Std Dev", 1, 3, 2)

# --- Indicator Calculation ---
df['ema_fast'] = ta.trend.ema_indicator(df['close'], fast_ema)
df['ema_slow'] = ta.trend.ema_indicator(df['close'], slow_ema)
df['rsi'] = ta.momentum.rsi(df['close'], rsi_period)
vwap_calc = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'])
df['vwap'] = vwap_calc.volume_weighted_average_price()
bb = ta.volatility.BollingerBands(df['close'], bb_period, bb_std)
df['bb_upper'] = bb.bollinger_hband()
df['bb_lower'] = bb.bollinger_lband()
df['volume_avg'] = df['volume'].rolling(10).mean()

# --- Signal Logic ---
df['buy'] = (
    (df['close'] > df['vwap']) &
    (df['ema_fast'] > df['ema_slow']) &
    (df['rsi'].between(40, 60)) &
    (df['close'] < df['bb_lower'] * 1.03) &
    (df['volume'] > df['volume_avg'])
)

df['sell'] = (
    (df['close'] < df['vwap']) &
    (df['ema_fast'] < df['ema_slow']) &
    ((df['rsi'] > 70) | (df['rsi'].diff() < -5)) &
    (df['close'] > df['bb_upper'] * 0.97) &
    (df['volume'] > df['volume_avg'])
)

df['buy_marker'] = np.where(df['buy'], df['low'] * 0.98, np.nan)
df['sell_marker'] = np.where(df['sell'], df['high'] * 1.02, np.nan)

# --- Signal Button ---
st.markdown("## 🧭 Current Trading Signal")
if df['buy'].iloc[-1]:
    st.success("🟢 Time to **BUY** based on indicators!")
elif df['sell'].iloc[-1]:
    st.error("🔴 Time to **SELL** based on indicators!")
else:
    st.info("🟡 No strong Buy/Sell signal right now.")

# --- Zoomed Candlestick Chart with Markers ---
st.markdown("## 📊 Zoomed Candlestick Chart with Signals")

apds = [
    mpf.make_addplot(df['ema_fast'], color='blue'),
    mpf.make_addplot(df['ema_slow'], color='orange'),
    mpf.make_addplot(df['vwap'], color='purple', linestyle=':'),
    mpf.make_addplot(df['bb_upper'], color='gray', linestyle='--'),
    mpf.make_addplot(df['bb_lower'], color='gray', linestyle='--'),
    mpf.make_addplot(df['buy_marker'], type='scatter', color='green', marker='^', markersize=100),
    mpf.make_addplot(df['sell_marker'], type='scatter', color='red', marker='v', markersize=100),
]

fig, _ = mpf.plot(
    df,
    type='candle',
    volume=True,
    addplot=apds,
    style='yahoo',
    title="Zoomed Technical Strategy Chart",
    figratio=(18, 10),
    figscale=2,
    returnfig=True
)
st.pyplot(fig)

# --- Indicator Visualization ---
st.markdown("## 📈 Indicator Trends")

fig, axs = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
axs[0].plot(df.index, df['rsi'], label='RSI', color='purple')
axs[0].axhline(70, color='red', linestyle='--')
axs[0].axhline(30, color='green', linestyle='--')
axs[0].legend()
axs[0].set_title("RSI")

axs[1].bar(df.index, df['volume'], label='Volume', alpha=0.4)
axs[1].plot(df.index, df['volume_avg'], label='10-period Avg Volume', color='orange')
axs[1].legend()
axs[1].set_title("Volume vs Avg")

st.pyplot(fig)

