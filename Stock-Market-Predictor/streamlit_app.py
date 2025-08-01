import streamlit as st
import pandas as pd
import ta
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

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

# Sidebar Configuration
st.sidebar.header("Settings")
st.sidebar.markdown("Configure the analysis parameters")

# File Upload
uploaded_file = st.sidebar.file_uploader(
    "Upload your stock data CSV file",
    type=["csv"],
    help="Upload a CSV file with columns: Date, Open, High, Low, Close, Volume"
)

# Technical Indicators Configuration
with st.sidebar.expander("Technical Indicators"):
    ema_short = st.slider("Short EMA Period", 5, 50, 20)
    ema_long = st.slider("Long EMA Period", 20, 200, 50)
    bb_period = st.slider("Bollinger Bands Period", 10, 50, 20)
    bb_std = st.slider("Bollinger Bands Std Dev", 1, 3, 2)

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
        df = pd.read_csv(uploaded_file)
        
        # Clean column names
        df.columns = [col.strip().title() for col in df.columns]
        
        # Standardize column names
        column_mapping = {
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
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
            window=14
        ).rsi()
        
        # MACD
        macd = ta.trend.MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        
        # Display indicators
        st.dataframe(df[[
            'Close', 
            f'EMA{ema_short}', 
            f'EMA{ema_long}', 
            'VWAP', 
            'BB_upper', 
            'BB_middle', 
            'BB_lower',
            'RSI',
            'MACD',
            'MACD_signal'
        ]].tail(10))
        
        # Generate trading signals
        st.subheader("Trading Signals")
        
        # Buy condition: Short EMA > Long EMA and Close > VWAP and RSI > 30
        df['Buy'] = (
            (df[f'EMA{ema_short}'] > df[f'EMA{ema_long}']) & 
            (df['Close'] > df['VWAP']) & 
            (df['RSI'] > 30)
        )
        
        # Sell condition: Short EMA < Long EMA and Close < VWAP and RSI < 70
        df['Sell'] = (
            (df[f'EMA{ema_short}'] < df[f'EMA{ema_long}']) & 
            (df['Close'] < df['VWAP']) & 
            (df['RSI'] < 70)
        )
        
        # Create markers for the chart
        df['Buy_Marker'] = df['Low'][df['Buy']] * 0.98
        df['Sell_Marker'] = df['High'][df['Sell']] * 1.02
        
        # Display signals
        st.write("Recent Buy Signals:")
        st.dataframe(df[df['Buy']].tail())
        
        st.write("Recent Sell Signals:")
        st.dataframe(df[df['Sell']].tail())
        
        # Visualization
        st.subheader("Interactive Chart")
        
        # Create plots
        add_plots = [
            mpf.make_addplot(df[f'EMA{ema_short}'], color='blue', width=1.5),
            mpf.make_addplot(df[f'EMA{ema_long}'], color='orange', width=1.5),
            mpf.make_addplot(df['VWAP'], color='purple', linestyle=':', width=1.5),
            mpf.make_addplot(df['BB_upper'], color='gray', linestyle='--', width=1),
            mpf.make_addplot(df['BB_lower'], color='gray', linestyle='--', width=1),
            mpf.make_addplot(df['Buy_Marker'], type='scatter', marker='^', 
                            markersize=100, color='green'),
            mpf.make_addplot(df['Sell_Marker'], type='scatter', marker='v', 
                            markersize=100, color='red')
        ]
        
        # Create figure
        fig, axes = mpf.plot(
            df,
            type='candle',
            volume=True,
            style='yahoo',
            title=f"Stock Analysis with Technical Indicators",
            addplot=add_plots,
            figscale=1.5,
            figratio=(12, 6),
            returnfig=True
        )
        
        # Display the figure in Streamlit
        st.pyplot(fig)
        
        # Additional Analysis
        st.subheader("Additional Analysis")
        
        # RSI Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**RSI Analysis**")
            st.write(f"Current RSI: {df['RSI'].iloc[-1]:.2f}")
            if df['RSI'].iloc[-1] > 70:
                st.warning("Overbought (RSI > 70)")
            elif df['RSI'].iloc[-1] < 30:
                st.success("Oversold (RSI < 30)")
            else:
                st.info("RSI in neutral range (30-70)")
            
            # RSI Chart
            fig_rsi, ax_rsi = plt.subplots(figsize=(10, 4))
            ax_rsi.plot(df.index, df['RSI'], label='RSI', color='purple')
            ax_rsi.axhline(70, color='red', linestyle='--')
            ax_rsi.axhline(30, color='green', linestyle='--')
            ax_rsi.set_title('RSI (14-day)')
            ax_rsi.legend()
            st.pyplot(fig_rsi)
        
        with col2:
            st.markdown("**MACD Analysis**")
            st.write(f"Current MACD: {df['MACD'].iloc[-1]:.2f}")
            st.write(f"Signal Line: {df['MACD_signal'].iloc[-1]:.2f}")
            
            if df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1]:
                st.success("Bullish (MACD above Signal)")
            else:
                st.warning("Bearish (MACD below Signal)")
            
            # MACD Chart
            fig_macd, ax_macd = plt.subplots(figsize=(10, 4))
            ax_macd.plot(df.index, df['MACD'], label='MACD', color='blue')
            ax_macd.plot(df.index, df['MACD_signal'], label='Signal', color='orange')
            ax_macd.bar(df.index, df['MACD_diff'], label='Histogram', color=np.where(df['MACD_diff'] > 0, 'g', 'r'))
            ax_macd.set_title('MACD (12,26,9)')
            ax_macd.legend()
            st.pyplot(fig_macd)
        
        # Performance Metrics
        st.subheader("Performance Metrics")
        
        # Calculate hypothetical returns based on signals
        if 'Buy' in df.columns and 'Sell' in df.columns:
            positions = []
            current_position = None
            entry_price = 0
            
            for i in range(len(df)):
                if df['Buy'].iloc[i] and current_position is None:
                    current_position = 'long'
                    entry_price = df['Close'].iloc[i]
                elif df['Sell'].iloc[i] and current_position == 'long':
                    exit_price = df['Close'].iloc[i]
                    positions.append({
                        'Entry Date': df.index[i - 1],
                        'Exit Date': df.index[i],
                        'Entry Price': entry_price,
                        'Exit Price': exit_price,
                        'Return': (exit_price - entry_price) / entry_price * 100
                    })
                    current_position = None
            
            if positions:
                trades_df = pd.DataFrame(positions)
                avg_return = trades_df['Return'].mean()
                win_rate = (trades_df['Return'] > 0).mean() * 100
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Trades", len(trades_df))
                col2.metric("Average Return (%)", f"{avg_return:.2f}%")
                col3.metric("Win Rate (%)", f"{win_rate:.2f}%")
                
                st.dataframe(trades_df.sort_values('Exit Date', ascending=False))
            else:
                st.warning("No completed trades found in the selected date range.")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload a CSV file to begin analysis.")
