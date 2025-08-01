import streamlit as st
import pandas as pd
import pandas_ta as ta
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
np.NaN


st.set_page_config(layout="wide")

st.title("\U0001F4C8 Stock Market Predictor")
st.write("Upload your stock CSV file with columns: Date, Open, High, Low, Close, Volume. Indicators: VWAP, EMA, Bollinger Bands, RSI. Signals: Buy/Sell with visual markers and market sentiment.")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

# Sidebar indicator settings
fast_ema = st.sidebar.slider("Fast EMA", 5, 30, 10)
slow_ema = st.sidebar.slider("Slow EMA", 20, 100, 50)
bb_period = st.sidebar.slider("Bollinger Band Period", 10, 50, 20)
bb_std = st.sidebar.slider("BB Std Dev", 1, 3, 2)
rsi_period = st.sidebar.slider("RSI Period", 7, 21, 14)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Ensure necessary columns exist
    required = {'date','open','high','low','close','volume'}
    if not required.issubset(df.columns):
        st.error(f"Missing required columns. Found: {df.columns.tolist()}")
    else:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df.dropna(subset=['date'], inplace=True)
        df.set_index('date', inplace=True)

        # Technical Indicators
        df['ema_fast'] = ta.ema(df['close'], length=fast_ema)
        df['ema_slow'] = ta.ema(df['close'], length=slow_ema)
        df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        df['rsi'] = ta.rsi(df['close'], length=rsi_period)

        bb = ta.bbands(df['close'], length=bb_period, std=bb_std)
        df['bb_upper'] = bb['BBU_'+str(bb_period)+'_'+str(bb_std).rstrip('0').rstrip('.')]
        df['bb_lower'] = bb['BBL_'+str(bb_period)+'_'+str(bb_std).rstrip('0').rstrip('.')]

        df['volume_avg'] = df['volume'].rolling(10).mean()

        # Signal Logic
        df['buy_signal'] = (
            (df['close'] > df['vwap']) &
            (df['ema_fast'] > df['ema_slow']) &
            (df['rsi'] > 40) & (df['rsi'] < 60) &
            (df['close'] <= df['bb_lower']) &
            (df['volume'] > df['volume_avg'])
        )

        df['sell_signal'] = (
            (df['close'] < df['vwap']) &
            (df['ema_fast'] < df['ema_slow']) &
            ((df['rsi'] > 70) | (df['rsi'].diff() < -5)) &
            (df['close'] >= df['bb_upper']) &
            (df['volume'] > df['volume_avg'])
        )

        # Date range filtering
        date_range = st.sidebar.date_input("Select Date Range", [df.index.min(), df.index.max()])
        if isinstance(date_range, list) and len(date_range) == 2:
            start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            df = df.loc[(df.index >= start_date) & (df.index <= end_date)]

        if df.empty:
            st.warning("\u26A0\uFE0F No data available for selected range.")
        else:
            # Chart plotting
            apds = [
                mpf.make_addplot(df['ema_fast'], color='green'),
                mpf.make_addplot(df['ema_slow'], color='red'),
                mpf.make_addplot(df['vwap'], color='blue'),
                mpf.make_addplot(df['bb_upper'], linestyle='--', color='gray'),
                mpf.make_addplot(df['bb_lower'], linestyle='--', color='gray'),
                mpf.make_addplot(df['rsi'], panel=1, color='purple'),
                mpf.make_addplot(df['volume'], panel=2, color='orange'),
            ]

            buy_markers = df[df['buy_signal']]
            sell_markers = df[df['sell_signal']]

            if not buy_markers.empty:
                apds.append(mpf.make_addplot(buy_markers['close'], type='scatter', markersize=100, marker='^', color='lime'))
            if not sell_markers.empty:
                apds.append(mpf.make_addplot(sell_markers['close'], type='scatter', markersize=100, marker='v', color='red'))

            fig, _ = mpf.plot(
                df,
                type='candle',
                style='yahoo',
                addplot=apds,
                volume=False,
                figscale=2.0,
                figratio=(16, 9),
                returnfig=True,
                title="Trading Strategy with Signals"
            )

            st.pyplot(fig)

            # Sentiment status
            st.subheader("\U0001F4CA Market Sentiment")
            if not df['buy_signal'].dropna().empty and df['buy_signal'].iloc[-1]:
                st.success("\U0001F4C8 BUY signal active!")
            elif not df['sell_signal'].dropna().empty and df['sell_signal'].iloc[-1]:
                st.error("\U0001F4C9 SELL signal active!")
            else:
                st.info("\U0001F610 Neutral sentiment")
