import streamlit as st
import pandas as pd
import ta
import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="📈 Stock Chart Analyzer", layout="wide")
st.title("📈 Candlestick Chart with Buy/Sell Signals")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload CSV with stock data", type=["csv"])

# Date Range selection
with st.sidebar.expander("Date Range"):
    today = datetime.now().date()
    default_start = today - timedelta(days=180)
    start_date = st.date_input("Start Date", default_start)
    end_date = st.date_input("End Date", today)

# Indicator configuration
with st.sidebar.expander("Indicators"):
    ema_short = st.slider("Short EMA", 5, 50, 20)
    ema_long = st.slider("Long EMA", 20, 200, 50)
    bb_period = st.slider("Bollinger Period", 10, 50, 20)
    bb_std = st.slider("Bollinger STD Dev", 1, 3, 2)
    rsi_period = st.slider("RSI Period", 5, 30, 14)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()

    if 'date' not in df.columns or not all(x in df.columns for x in ['open', 'high', 'low', 'close', 'volume']):
        st.error("CSV must contain columns: Date, Open, High, Low, Close, Volume")
        st.stop()

    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    df_range = df[(df.index.date >= start_date) & (df.index.date <= end_date)].copy()

    if df_range.empty:
        st.warning("⚠️ No data available for the selected date range.")
        st.stop()

    df_range['vwap'] = ta.volume.VolumeWeightedAveragePrice(df_range['high'], df_range['low'], df_range['close'], df_range['volume']).volume_weighted_average_price()
    df_range[f'ema{ema_short}'] = ta.trend.EMAIndicator(df_range['close'], ema_short).ema_indicator()
    df_range[f'ema{ema_long}'] = ta.trend.EMAIndicator(df_range['close'], ema_long).ema_indicator()

    bb = ta.volatility.BollingerBands(df_range['close'], bb_period, bb_std)
    df_range['bb_upper'] = bb.bollinger_hband()
    df_range['bb_lower'] = bb.bollinger_lband()

    macd = ta.trend.MACD(df_range['close'])
    df_range['macd'] = macd.macd()
    df_range['macd_signal'] = macd.macd_signal()

    df_range['rsi'] = ta.momentum.RSIIndicator(df_range['close'], rsi_period).rsi()

    # Buy/Sell Conditions
    df_range['buy'] = (df_range[f'ema{ema_short}'] > df_range[f'ema{ema_long}']) & \
                      (df_range['close'] > df_range['vwap']) & \
                      (df_range['macd'] > df_range['macd_signal']) & \
                      (df_range['rsi'] > 30)

    df_range['sell'] = (df_range[f'ema{ema_short}'] < df_range[f'ema{ema_long}']) & \
                       (df_range['close'] < df_range['vwap']) & \
                       (df_range['macd'] < df_range['macd_signal']) & \
                       (df_range['rsi'] < 70)

    df_range['buy_marker'] = np.where(df_range['buy'].diff() == 1, df_range['low'] * 0.98, np.nan)
    df_range['sell_marker'] = np.where(df_range['sell'].diff() == 1, df_range['high'] * 1.02, np.nan)

    # Show last signals
    st.subheader("Recent Buy/Sell Signals")
    col1, col2 = st.columns(2)
    col1.write(df_range[df_range['buy_marker'].notna()].tail(5))
    col2.write(df_range[df_range['sell_marker'].notna()].tail(5))

    # Plot chart
    st.subheader("📊 Candlestick Chart")

    apds = [
        mpf.make_addplot(df_range[f'ema{ema_short}'], color='blue', width=1.2),
        mpf.make_addplot(df_range[f'ema{ema_long}'], color='orange', width=1.2),
        mpf.make_addplot(df_range['vwap'], color='purple', linestyle='--'),
        mpf.make_addplot(df_range['bb_upper'], color='gray', linestyle='--'),
        mpf.make_addplot(df_range['bb_lower'], color='gray', linestyle='--'),
        mpf.make_addplot(df_range['buy_marker'], type='scatter', marker='^', markersize=100, color='green'),
        mpf.make_addplot(df_range['sell_marker'], type='scatter', marker='v', markersize=100, color='red')
    ]

    fig, axes = mpf.plot(
        df_range,
        type='candle',
        volume=True,
        style='yahoo',
        title="Strategy Chart (VWAP, EMA, Bollinger Bands, Buy/Sell)",
        addplot=apds,
        figscale=1.8,
        figratio=(16, 9),
        returnfig=True
    )
    st.pyplot(fig)

else:
    st.info("Upload your CSV file from NSE or Yahoo Finance to begin analysis.")
    # --- BUY/SELL Button Indicator at Top ---
if 'data' in st.session_state and st.session_state.data is not None:
    df_signal = st.session_state.data.copy()
    if 'buy' in df_signal.columns and 'sell' in df_signal.columns:
        signal = "Neutral"
        color = "gray"
        if df_signal['buy'].iloc[-1]:
            signal = "BUY NOW"
            color = "green"
        elif df_signal['sell'].iloc[-1]:
            signal = "SELL NOW"
            color = "red"
        
        st.markdown(f"""
        <div style='text-align:center; margin-top:10px; margin-bottom:10px;'>
            <button style="background-color:{color}; color:white; font-size:20px; padding:10px 30px; border:none; border-radius:10px;">
                {signal}
            </button>
        </div>
        """, unsafe_allow_html=True)

