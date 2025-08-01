import streamlit as st
import pandas as pd
import ta
import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Stock Signal Analyzer",
    layout="wide"
)

st.title("📈 Advanced Stock Signal Chart with Zoom and Indicators")
st.markdown("Upload stock data and visualize buy/sell signals based on advanced technical logic.")

# Upload CSV
df_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if df_file:
    df = pd.read_csv(df_file)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Parse date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df.dropna(subset=['date'], inplace=True)
        df.set_index('date', inplace=True)
    else:
        st.error("No 'Date' column found.")
        st.stop()

    # Filter date range
    with st.sidebar.expander("Select Date Range"):
        start_date = st.date_input("Start Date", df.index.min().date())
        end_date = st.date_input("End Date", df.index.max().date())
        df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]

    # Dropna & cast
    df = df[['open', 'high', 'low', 'close', 'volume']].dropna()

    # Calculate indicators
    df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
    df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=10)
    df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=50)
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['vol_ma10'] = df['volume'].rolling(10).mean()

    # Define buy and sell logic
    df['buy'] = (
        (df['close'] > df['vwap']) &
        (df['ema_fast'] > df['ema_slow']) &
        (df['rsi'] > 40) & (df['rsi'] < 60) &
        (df['close'] < df['bb_lower'] * 1.03) &
        (df['volume'] > df['vol_ma10'])
    )

    df['sell'] = (
        (df['close'] < df['vwap']) &
        (df['ema_fast'] < df['ema_slow']) &
        ((df['rsi'] > 70) | (df['rsi'].diff() < -10)) &
        (df['close'] > df['bb_upper'] * 0.97) &
        (df['volume'] > df['vol_ma10'])
    )

    df['buy_marker'] = np.where(df['buy'], df['low'] * 0.98, np.nan)
    df['sell_marker'] = np.where(df['sell'], df['high'] * 1.02, np.nan)

    # Display current signal
    latest_signal = "Neutral"
    if df['buy'].iloc[-1]:
        latest_signal = "📥 BUY"
    elif df['sell'].iloc[-1]:
        latest_signal = "📤 SELL"
    st.markdown(f"### **Current Signal: {latest_signal}**")

    # Prepare mplfinance plots
    apds = [
        mpf.make_addplot(df['ema_fast'], color='blue'),
        mpf.make_addplot(df['ema_slow'], color='orange'),
        mpf.make_addplot(df['vwap'], color='purple', linestyle='--'),
        mpf.make_addplot(df['bb_upper'], color='gray', linestyle='--'),
        mpf.make_addplot(df['bb_lower'], color='gray', linestyle='--')
    ]

    if df['buy_marker'].dropna().shape[0] > 0:
        apds.append(mpf.make_addplot(df['buy_marker'], type='scatter', marker='^', color='green', markersize=200))
    if df['sell_marker'].dropna().shape[0] > 0:
        apds.append(mpf.make_addplot(df['sell_marker'], type='scatter', marker='v', color='red', markersize=200))

    # Plot chart
    if not df.empty:
        st.subheader("Zoomed Candlestick Chart with Signals")
        fig, _ = mpf.plot(
            df,
            type='candle',
            volume=True,
            style='yahoo',
            title='Zoomed Strategy Chart (Buy/Sell Signals)',
            figscale=2.5,
            figratio=(18, 9),
            addplot=apds,
            returnfig=True,
            tight_layout=True
        )
        st.pyplot(fig)
    else:
        st.warning("No data to plot in selected date range.")

else:
    st.info("Please upload a CSV file to begin.")
    st.markdown("""
    #### Format Required:
    - Columns: `Date, Open, High, Low, Close, Volume`
    - Date should be in `DD-MM-YYYY` or similar format
    """)
