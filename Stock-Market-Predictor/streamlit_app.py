import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import mplfinance as mpf
from datetime import datetime

st.set_page_config(page_title="Stock Strategy Chart", layout="wide")
st.title("📈 Strategy Chart with VWAP, EMA & Bollinger Bands")

# Load CSV file
uploaded_file = st.file_uploader("Upload your stock CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Convert all column names to lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Check required columns
    required = ['date', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required):
        st.error(f"CSV must contain: {', '.join(required)}")
        st.stop()

    # Parse date
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df.set_index('date', inplace=True)

    # Drop any rows with missing core data
    df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)

    # Calculate Indicators
    df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)

    df['vwap'] = ta.volume.VolumeWeightedAveragePrice(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        volume=df['volume']
    ).volume_weighted_average_price()

    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()

    # Buy/Sell markers (dummy logic – can be improved)
    df['buy_marker'] = np.where((df['close'] > df['ema20']) & (df['close'] > df['ema50']), df['low'] * 0.98, np.nan)
    df['sell_marker'] = np.where((df['close'] < df['ema20']) & (df['close'] < df['ema50']), df['high'] * 1.02, np.nan)

    # Show table
    st.subheader("Data Snapshot")
    st.dataframe(df.tail())

    # Plot chart
    apds = [
        mpf.make_addplot(df['ema20'], color='blue', width=1.5),
        mpf.make_addplot(df['ema50'], color='orange', width=1.5),
        mpf.make_addplot(df['vwap'], color='purple', linestyle='--'),
        mpf.make_addplot(df['bb_upper'], color='gray', linestyle='--'),
        mpf.make_addplot(df['bb_lower'], color='gray', linestyle='--'),
        mpf.make_addplot(df['buy_marker'], type='scatter', marker='^', color='green', markersize=100),
        mpf.make_addplot(df['sell_marker'], type='scatter', marker='v', color='red', markersize=100)
    ]

    fig, axlist = mpf.plot(
        df,
        type='candle',
        volume=True,
        style='yahoo',
        title="Strategy Chart (VWAP, EMA, Bollinger Bands)",
        addplot=apds,
        figscale=2.0,
        figratio=(18, 10),
        returnfig=True,
        savefig='figure1_strategy_chart.png'
    )

    st.subheader("Candlestick Chart with Indicators")
    st.pyplot(fig)
    st.success("Chart saved as 'figure1_strategy_chart.png'")
else:
    st.info("Please upload a CSV file with columns: Date, Open, High, Low, Close, Volume")
