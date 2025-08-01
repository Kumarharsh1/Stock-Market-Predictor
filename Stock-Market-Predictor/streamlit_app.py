import streamlit as st
import pandas as pd
import ta
import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# App Configuration
st.set_page_config(
    page_title="Stock Market Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and Description
st.title("📈 Stock Market Predictor")
st.markdown("""
This app analyzes stock market data using technical indicators and provides buy/sell signals.
Upload your stock data CSV file to get started.
""")

# Initialize session state
if 'sample_loaded' not in st.session_state:
    st.session_state.sample_loaded = False

# Sample Data Button
if st.button("Load Sample Data (ADANIPORTS)"):
    try:
        sample_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/stocks.csv"
        sample_data = pd.read_csv(sample_url)
        sample_data.to_csv("sample_stock_data.csv", index=False)
        st.session_state.sample_loaded = True
        st.rerun()
    except Exception as e:
        st.error(f"Couldn't load sample data: {str(e)}")

# Sidebar Configuration
st.sidebar.header("Settings")
st.sidebar.markdown("Configure the analysis parameters")

# File Upload
uploaded_file = st.sidebar.file_uploader(
    "Upload your stock data CSV file",
    type=["csv"],
    help="Upload a CSV file with columns: Date, Open, High, Low, Close, Volume"
)

# Use sample data if no file uploaded but sample was requested
if uploaded_file is None and st.session_state.sample_loaded:
    try:
        uploaded_file = "sample_stock_data.csv"
    except:
        st.warning("Sample data could not be loaded")

# Technical Indicators Configuration
with st.sidebar.expander("Technical Indicators"):
    ema_short = st.slider("Short EMA Period", 5, 50, 20)
    ema_long = st.slider("Long EMA Period", 20, 200, 50)
    bb_period = st.slider("Bollinger Bands Period", 10, 50, 20)
    bb_std = st.slider("Bollinger Bands Std Dev", 1, 3, 2)
    rsi_period = st.slider("RSI Period", 5, 30, 14)

# Date Range Configuration
with st.sidebar.expander("Date Range"):
    default_end = datetime.now()
    default_start = default_end - timedelta(days=365)
    start_date = st.date_input("Start Date", default_start)
    end_date = st.date_input("End Date", default_end)

# Main App Functionality
if uploaded_file is not None:
    try:
        # Load and preprocess data
        if isinstance(uploaded_file, str):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        
        # Clean column names
        df.columns = [col.strip().title() for col in df.columns]
        
        # Standardize column names
        column_mapping = {
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        # Find matching columns (case insensitive)
        final_mapping = {}
        for standard_name, possible_names in column_mapping.items():
            for col in df.columns:
                if col.lower() == possible_names.lower():
                    final_mapping[col] = standard_name
        
        df = df.rename(columns=final_mapping)
        
        # Convert date and filter by date range
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[(df['Date'] >= pd.to_datetime(start_date)) & 
                (df['Date'] <= pd.to_datetime(end_date))]
        
        # Set Date as index
        df.set_index('Date', inplace=True)
        
        # Drop rows with missing values
        df.dropna(inplace=True)
        
        # Display raw data
        with st.expander("View Raw Data"):
            st.dataframe(df)
        
        # Calculate Technical Indicators
        st.subheader("Technical Indicators")
        
        # VWAP
        df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(
            high=df['High'], 
            low=df['Low'], 
            close=df['Close'], 
            volume=df['Volume']
        ).volume_weighted_average_price()
        
        # EMAs
        df[f'EMA{ema_short}'] = ta.trend.EMAIndicator(
            close=df['Close'], 
            window=ema_short
        ).ema_indicator()
        
        df[f'EMA{ema_long}'] = ta.trend.EMAIndicator(
            close=df['Close'], 
            window=ema_long
        ).ema_indicator()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(
            close=df['Close'], 
            window=bb_period, 
            window_dev=bb_std
        )
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
        
        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(
            close=df['Close'], 
            window=rsi_period
        ).rsi()
        
        # MACD
        macd = ta.trend.MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        
        # Generate trading signals
        st.subheader("Trading Signals")
        
        # Buy condition
        df['Buy'] = (
            (df[f'EMA{ema_short}'] > df[f'EMA{ema_long}']) & 
            (df['Close'] > df['VWAP']) & 
            (df['RSI'] > 30) &
            (df['MACD'] > df['MACD_signal'])
        )
        
        # Sell condition
        df['Sell'] = (
            (df[f'EMA{ema_short}'] < df[f'EMA{ema_long}']) & 
            (df['Close'] < df['VWAP']) & 
            (df['RSI'] < 70) &
            (df['MACD'] < df['MACD_signal'])
        )
        
        # Create markers for the chart
        df['Buy_Marker'] = np.where(df['Buy'], df['Low'] * 0.98, np.nan)
        df['Sell_Marker'] = np.where(df['Sell'], df['High'] * 1.02, np.nan)
        
        # Display signals
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Recent Buy Signals**")
            buy_signals = df[df['Buy']].tail(5)
            if not buy_signals.empty:
                st.dataframe(buy_signals[['Close', f'EMA{ema_short}', f'EMA{ema_long}', 'VWAP', 'RSI']])
            else:
                st.warning("No buy signals detected in this period")
        
        with col2:
            st.write("**Recent Sell Signals**")
            sell_signals = df[df['Sell']].tail(5)
            if not sell_signals.empty:
                st.dataframe(sell_signals[['Close', f'EMA{ema_short}', f'EMA{ema_long}', 'VWAP', 'RSI']])
            else:
                st.warning("No sell signals detected in this period")
        
        # Visualization
        st.subheader("Interactive Chart")
        
        # Create plots
        apds = [
            mpf.make_addplot(df[f'EMA{ema_short}'], color='blue', width=1.5, panel=0),
            mpf.make_addplot(df[f'EMA{ema_long}'], color='orange', width=1.5, panel=0),
            mpf.make_addplot(df['VWAP'], color='purple', linestyle=':', width=1.5, panel=0),
            mpf.make_addplot(df['BB_upper'], color='gray', linestyle='--', width=1, panel=0),
            mpf.make_addplot(df['BB_lower'], color='gray', linestyle='--', width=1, panel=0),
            mpf.make_addplot(df['Buy_Marker'], type='scatter', marker='^', 
                            markersize=100, color='green', panel=0),
            mpf.make_addplot(df['Sell_Marker'], type='scatter', marker='v', 
                            markersize=100, color='red', panel=0)
        ]
        
        # Create figure
        fig, axes = mpf.plot(
            df,
            type='candle',
            volume=True,
            style='yahoo',
            title=f"Stock Analysis with Technical Indicators",
            addplot=apds,
            figscale=1.5,
            figratio=(12, 6),
            returnfig=True
        )
        
        # Display the figure in Streamlit
        st.pyplot(fig)
        
        # Additional Analysis
        st.subheader("Market Sentiment Analysis")
        
        # Bullish/Bearish Signal
        last_signal = "Neutral"
        last_color = "gray"
        
        if df['Buy'].iloc[-1]:
            last_signal = "Bullish"
            last_color = "green"
        elif df['Sell'].iloc[-1]:
            last_signal = "Bearish"
            last_color = "red"
        
        # Create three columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Signal", last_signal, delta_color="off")
            st.markdown(f"""<div style='background-color:{last_color};padding:10px;border-radius:5px;text-align:center;color:white'>
                            {last_signal} Conditions</div>""", unsafe_allow_html=True)
        
        with col2:
            st.metric("RSI Value", f"{df['RSI'].iloc[-1]:.2f}")
            if df['RSI'].iloc[-1] > 70:
                st.error("Overbought Territory")
            elif df['RSI'].iloc[-1] < 30:
                st.success("Oversold Territory")
            else:
                st.info("Neutral Territory")
        
        with col3:
            macd_diff = df['MACD'].iloc[-1] - df['MACD_signal'].iloc[-1]
            st.metric("MACD Difference", f"{macd_diff:.4f}")
            if macd_diff > 0:
                st.success("Bullish Momentum")
            else:
                st.warning("Bearish Momentum")
        
        # RSI and MACD Charts
        st.subheader("Indicator Charts")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # RSI Chart
        ax1.plot(df.index, df['RSI'], label='RSI', color='purple')
        ax1.axhline(70, color='red', linestyle='--', alpha=0.5)
        ax1.axhline(30, color='green', linestyle='--', alpha=0.5)
        ax1.fill_between(df.index, 70, df['RSI'], where=(df['RSI']>=70), color='red', alpha=0.2)
        ax1.fill_between(df.index, 30, df['RSI'], where=(df['RSI']<=30), color='green', alpha=0.2)
        ax1.set_title('RSI (14-day)')
        ax1.legend()
        
        # MACD Chart
        ax2.plot(df.index, df['MACD'], label='MACD', color='blue')
        ax2.plot(df.index, df['MACD_signal'], label='Signal Line', color='orange')
        ax2.bar(df.index, df['MACD_diff'], 
               color=np.where(df['MACD_diff']>0, 'green', 'red'), 
               label='Histogram', alpha=0.5)
        ax2.set_title('MACD (12,26,9)')
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Performance Metrics
        st.subheader("Strategy Performance")
        
        # Calculate hypothetical returns based on signals
        if 'Buy' in df.columns and 'Sell' in df.columns:
            positions = []
            current_position = None
            entry_price = 0
            
            for i in range(len(df)):
                if df['Buy'].iloc[i] and current_position is None:
                    current_position = 'long'
                    entry_price = df['Close'].iloc[i]
                    entry_date = df.index[i]
                elif df['Sell'].iloc[i] and current_position == 'long':
                    exit_price = df['Close'].iloc[i]
                    exit_date = df.index[i]
                    positions.append({
                        'Entry Date': entry_date,
                        'Exit Date': exit_date,
                        'Entry Price': entry_price,
                        'Exit Price': exit_price,
                        'Return (%)': (exit_price - entry_price) / entry_price * 100,
                        'Holding Days': (exit_date - entry_date).days
                    })
                    current_position = None
            
            if positions:
                trades_df = pd.DataFrame(positions)
                avg_return = trades_df['Return (%)'].mean()
                win_rate = (trades_df['Return (%)'] > 0).mean() * 100
                total_return = trades_df['Return (%)'].sum()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Trades", len(trades_df))
                col2.metric("Average Return (%)", f"{avg_return:.2f}%")
                col3.metric("Win Rate (%)", f"{win_rate:.2f}%")
                col4.metric("Total Strategy Return (%)", f"{total_return:.2f}%")
                
                st.write("Trade History:")
                st.dataframe(trades_df.sort_values('Exit Date', ascending=False))
                
                # Plot cumulative returns
                cumulative_returns = trades_df['Return (%)'].cumsum()
                fig_returns, ax_returns = plt.subplots(figsize=(10, 4))
                ax_returns.plot(trades_df['Exit Date'], cumulative_returns, marker='o')
                ax_returns.set_title("Cumulative Returns Over Time")
                ax_returns.set_ylabel("Return (%)")
                ax_returns.grid(True)
                st.pyplot(fig_returns)
            else:
                st.warning("No completed trades found in the selected date range.")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your CSV file format and try again.")
else:
    st.info("Please upload a CSV file or click 'Load Sample Data' to begin analysis.")
    st.markdown("""
    ### Expected CSV Format:
    ```
    Date,Open,High,Low,Close,Volume
    2023-01-01,150.2,152.5,149.8,151.3,2500000
    2023-01-02,151.5,153.1,150.9,152.4,1800000
    ...
    ```
    """)
