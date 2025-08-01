import streamlit as st
import pandas as pd
import ta
import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import os

# App Configuration
st.set_page_config(
    page_title="📈 Stock Market Predictor",
    page_icon="📉",
    layout="wide"
)

# Title and Description
st.title("📈 Stock Market Predictor")
st.markdown("Analyze stock data, detect Buy/Sell signals using technical indicators, and visualize market sentiment.")

# Show files (for debugging only)
# st.write("Visible files:", os.listdir())

# Session State
if 'sample_loaded' not in st.session_state:
    st.session_state.sample_loaded = False

# Load Sample Data
if st.button("📂 Load Sample Data (ADANIPORTS)"):
    try:
        sample_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/stocks.csv"
        sample_data = pd.read_csv(sample_url)
        sample_data.rename(columns={"close": "Close", "date": "Date"}, inplace=True)
        sample_data["Open"] = sample_data["Close"]
        sample_data["High"] = sample_data["Close"]
        sample_data["Low"] = sample_data["Close"]
        sample_data["Volume"] = 1000000
        sample_data.to_csv("sample_stock_data.csv", index=False)
        st.session_state.sample_loaded = True
        st.rerun()
    except Exception as e:
        st.error(f"Sample data load failed: {e}")

# Sidebar
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Date, Open, High, Low, Close, Volume)", type=["csv"])

if uploaded_file is None and st.session_state.sample_loaded:
    uploaded_file = "sample_stock_data.csv"

# Indicator Settings
with st.sidebar.expander("📊 Technical Indicators"):
    ema_short = st.slider("Short EMA", 5, 50, 20)
    ema_long = st.slider("Long EMA", 20, 200, 50)
    bb_period = st.slider("Bollinger Period", 10, 50, 20)
    bb_std = st.slider("Bollinger Std Dev", 1, 3, 2)
    rsi_period = st.slider("RSI Period", 5, 30, 14)

# Date Filter
with st.sidebar.expander("📅 Date Range"):
    default_end = datetime.now()
    default_start = default_end - timedelta(days=365)
    start_date = st.date_input("Start Date", default_start)
    end_date = st.date_input("End Date", default_end)

# If file exists, start processing
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.title()

        column_mapping = {
            'Date': 'Date', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 
            'Close': 'Close', 'Volume': 'Volume'
        }

        df.rename(columns=column_mapping, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
        df.set_index('Date', inplace=True)
        df.dropna(inplace=True)

        # Show raw data
        with st.expander("🧾 Raw Data"):
            st.dataframe(df)

        # Indicators
        df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume']).volume_weighted_average_price()
        df[f'EMA{ema_short}'] = ta.trend.EMAIndicator(df['Close'], ema_short).ema_indicator()
        df[f'EMA{ema_long}'] = ta.trend.EMAIndicator(df['Close'], ema_long).ema_indicator()
        bb = ta.volatility.BollingerBands(df['Close'], bb_period, bb_std)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], rsi_period).rsi()
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()

        # Signals
        df['Buy'] = (df[f'EMA{ema_short}'] > df[f'EMA{ema_long}']) & (df['Close'] > df['VWAP']) & (df['RSI'] > 30) & (df['MACD'] > df['MACD_signal'])
        df['Sell'] = (df[f'EMA{ema_short}'] < df[f'EMA{ema_long}']) & (df['Close'] < df['VWAP']) & (df['RSI'] < 70) & (df['MACD'] < df['MACD_signal'])
        df['Buy_Marker'] = np.where(df['Buy'], df['Low'] * 0.98, np.nan)
        df['Sell_Marker'] = np.where(df['Sell'], df['High'] * 1.02, np.nan)

        # Recent Signals
        col1, col2 = st.columns(2)
        with col1:
            st.write("📈 **Recent Buy Signals**")
            st.dataframe(df[df['Buy']].tail(5)[['Close', f'EMA{ema_short}', f'EMA{ema_long}', 'RSI']])
        with col2:
            st.write("📉 **Recent Sell Signals**")
            st.dataframe(df[df['Sell']].tail(5)[['Close', f'EMA{ema_short}', f'EMA{ema_long}', 'RSI']])

        # Chart
        st.subheader("📉 Candlestick Chart with Signals")
        apds = [
            mpf.make_addplot(df[f'EMA{ema_short}'], color='blue'),
            mpf.make_addplot(df[f'EMA{ema_long}'], color='orange'),
            mpf.make_addplot(df['VWAP'], color='purple', linestyle=':'),
            mpf.make_addplot(df['BB_upper'], color='gray', linestyle='--'),
            mpf.make_addplot(df['BB_lower'], color='gray', linestyle='--'),
            mpf.make_addplot(df['Buy_Marker'], type='scatter', marker='^', color='green', markersize=100),
            mpf.make_addplot(df['Sell_Marker'], type='scatter', marker='v', color='red', markersize=100),
        ]
        fig, _ = mpf.plot(df, type='candle', volume=True, style='yahoo', addplot=apds,
                          figscale=1.5, figratio=(12,6), returnfig=True)
        st.pyplot(fig)

        # Sentiment
        st.subheader("📌 Market Sentiment")
        last_signal = "Neutral"
        color = "gray"
        if df['Buy'].iloc[-1]:
            last_signal = "Bullish"
            color = "green"
        elif df['Sell'].iloc[-1]:
            last_signal = "Bearish"
            color = "red"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Signal", last_signal)
            st.markdown(f"<div style='background-color:{color};color:white;padding:10px;text-align:center;border-radius:5px'>{last_signal} Conditions</div>", unsafe_allow_html=True)
        with col2:
            rsi_val = df['RSI'].iloc[-1]
            st.metric("RSI", f"{rsi_val:.2f}")
            if rsi_val > 70:
                st.error("Overbought")
            elif rsi_val < 30:
                st.success("Oversold")
            else:
                st.info("Neutral")
        with col3:
            macd_diff = df['MACD_diff'].iloc[-1]
            st.metric("MACD Diff", f"{macd_diff:.2f}")
            if macd_diff > 0:
                st.success("Bullish Momentum")
            else:
                st.warning("Bearish Momentum")

        # Indicator Charts
        st.subheader("📈 RSI & MACD")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        ax1.plot(df.index, df['RSI'], label='RSI', color='purple')
        ax1.axhline(70, color='red', linestyle='--')
        ax1.axhline(30, color='green', linestyle='--')
        ax1.set_title("RSI")
        ax2.plot(df.index, df['MACD'], label='MACD', color='blue')
        ax2.plot(df.index, df['MACD_signal'], label='Signal', color='orange')
        ax2.bar(df.index, df['MACD_diff'], color=np.where(df['MACD_diff']>0, 'green', 'red'), alpha=0.5)
        ax2.set_title("MACD")
        st.pyplot(fig)

        # Strategy Evaluation
        st.subheader("📊 Strategy Performance")
        positions = []
        holding = None
        for i in range(len(df)):
            if df['Buy'].iloc[i] and holding is None:
                holding = {'entry_price': df['Close'].iloc[i], 'entry_date': df.index[i]}
            elif df['Sell'].iloc[i] and holding:
                exit_price = df['Close'].iloc[i]
                positions.append({
                    'Entry Date': holding['entry_date'],
                    'Exit Date': df.index[i],
                    'Entry Price': holding['entry_price'],
                    'Exit Price': exit_price,
                    'Return (%)': (exit_price - holding['entry_price']) / holding['entry_price'] * 100,
                    'Days': (df.index[i] - holding['entry_date']).days
                })
                holding = None
        if positions:
            trades_df = pd.DataFrame(positions)
            total_return = trades_df['Return (%)'].sum()
            avg_return = trades_df['Return (%)'].mean()
            win_rate = (trades_df['Return (%)'] > 0).mean() * 100
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Trades", len(trades_df))
            col2.metric("Avg Return (%)", f"{avg_return:.2f}")
            col3.metric("Win Rate (%)", f"{win_rate:.2f}")
            col4.metric("Total Return (%)", f"{total_return:.2f}")
            st.dataframe(trades_df)
            # Cumulative Returns
            fig_ret, ax_ret = plt.subplots(figsize=(10, 4))
            ax_ret.plot(trades_df['Exit Date'], trades_df['Return (%)'].cumsum(), marker='o')
            ax_ret.set_title("Cumulative Return")
            st.pyplot(fig_ret)
        else:
            st.warning("No trades completed in the selected period.")

    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.info("Upload a CSV file or click 'Load Sample Data' to begin.")
