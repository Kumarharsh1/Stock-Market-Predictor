import streamlit as st
import pandas as pd
import ta
import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(
    page_title="Stock Market Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Stock Market Predictor")
st.markdown("""
Analyze stock data, detect Buy/Sell signals using technical indicators, and make predictions with a machine learning model.
""")

# --- Sidebar for user input ---
st.sidebar.header("User Input")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker Symbol", value='AAPL').upper()

# --- Date Range from yfinance ---
default_end = datetime.now().date()
default_start = default_end - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", default_end)

if st.sidebar.button("Fetch Data"):
    if ticker_symbol:
        try:
            # Using yfinance to download data
            data = yf.download(ticker_symbol, start=start_date, end=end_date)
            if not data.empty:
                st.session_state.data = data
                st.session_state.ticker = ticker_symbol
                st.success(f"Successfully fetched data for {ticker_symbol}!")
            else:
                st.error(f"No data found for the ticker symbol: {ticker_symbol} in the selected date range. Please check the ticker and dates.")
                st.session_state.data = None
        except Exception as e:
            st.error(f"An error occurred while fetching data: {e}")
            st.session_state.data = None
    else:
        st.error("Please enter a valid stock ticker symbol.")

# --- Technical Indicators Settings ---
with st.sidebar.expander("Technical Indicators"):
    ema_short = st.slider("Short EMA Period", 5, 50, 20)
    ema_long = st.slider("Long EMA Period", 20, 200, 50)
    bb_period = st.slider("Bollinger Bands Period", 10, 50, 20)
    bb_std = st.slider("Bollinger Bands Std Dev", 1, 3, 2)
    rsi_period = st.slider("RSI Period", 5, 30, 14)

# --- Main App Logic ---
if 'data' in st.session_state and st.session_state.data is not None:
    df = st.session_state.data.copy()

    # --- CRITICAL FIX: Add this check to prevent errors with empty dataframes ---
    if df.empty:
        st.warning("The selected data is empty. Please check your ticker and date range.")
        st.stop()
    
    # Rename columns to lowercase for ta library
    df.columns = [col.lower() for col in df.columns]

    st.subheader(f"Analyzing Stock: {st.session_state.ticker}")
    st.write(df.tail())
    
    try:
        # --- Technical Indicator Calculation ---
        df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
        df[f'ema{ema_short}'] = ta.trend.EMAIndicator(df['close'], ema_short).ema_indicator()
        df[f'ema{ema_long}'] = ta.trend.EMAIndicator(df['close'], ema_long).ema_indicator()
        bb = ta.volatility.BollingerBands(df['close'], bb_period, bb_std)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], rsi_period).rsi()
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # --- Buy/Sell Signal Logic ---
        df['buy'] = (
            (df[f'ema{ema_short}'] > df[f'ema{ema_long}']) &
            (df['close'] > df['vwap']) &
            (df['rsi'] > 30) &
            (df['macd'] > df['macd_signal'])
        )
        df['sell'] = (
            (df[f'ema{ema_short}'] < df[f'ema{ema_long}']) &
            (df['close'] < df['vwap']) &
            (df['rsi'] < 70) &
            (df['macd'] < df['macd_signal'])
        )
        
        df['buy_marker'] = np.where(df['buy'].diff() == 1, df['low'] * 0.98, np.nan)
        df['sell_marker'] = np.where(df['sell'].diff() == 1, df['high'] * 1.02, np.nan)

        # --- Candlestick Chart with Signals ---
        st.subheader("Candlestick Chart with Signals")
        apds = [
            mpf.make_addplot(df[f'ema{ema_short}'], color='blue', width=1.5, panel=0),
            mpf.make_addplot(df[f'ema{ema_long}'], color='orange', width=1.5, panel=0),
            mpf.make_addplot(df['vwap'], color='purple', linestyle=':', width=1.5, panel=0),
            mpf.make_addplot(df['bb_upper'], color='gray', linestyle='--', width=1, panel=0),
            mpf.make_addplot(df['bb_lower'], color='gray', linestyle='--', width=1, panel=0),
            mpf.make_addplot(df['buy_marker'], type='scatter', marker='^', markersize=100, color='green', panel=0),
            mpf.make_addplot(df['sell_marker'], type='scatter', marker='v', markersize=100, color='red', panel=0)
        ]
        fig, axes = mpf.plot(
            df,
            type='candle',
            volume=True,
            style='yahoo',
            title=f"{st.session_state.ticker} Stock Analysis",
            addplot=apds,
            figscale=1.5,
            figratio=(12, 6),
            returnfig=True
        )
        st.pyplot(fig)
        
        # --- Machine Learning Prediction Section ---
        st.subheader("Machine Learning Prediction")
        
        # Create features and target for the model
        df['prediction_target'] = df['close'].shift(-1)
        df.dropna(inplace=True)
        
        features = df[['open', 'high', 'low', 'close', 'volume', 'vwap', f'ema{ema_short}', 'rsi', 'macd']]
        target = df['prediction_target']

        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Train a Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate model performance
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.markdown(f"""
        **Model Evaluation:**
        - Mean Squared Error (MSE): `{mse:.4f}`
        - R-squared (R²): `{r2:.4f}`
        """)

        # Predict the next day's closing price
        last_day_features = df[['open', 'high', 'low', 'close', 'volume', 'vwap', f'ema{ema_short}', 'rsi', 'macd']].iloc[-1].values.reshape(1, -1)
        next_day_prediction = model.predict(last_day_features)[0]

        col1, col2 = st.columns(2)
        col1.metric("Predicted Next Day's Close Price", f"${next_day_prediction:.2f}")
        col2.metric("Last Recorded Close Price", f"${df['close'].iloc[-1]:.2f}")
        
    except Exception as e:
        st.error(f"An error occurred during analysis or prediction: {e}")
        st.stop()
else:
    st.info("Please enter a stock ticker symbol and click 'Fetch Data' to begin analysis.")
    st.markdown("""
    ### About the App:
    - **Technical Analysis:** The app uses popular indicators like EMA, Bollinger Bands, RSI, and MACD to generate Buy/Sell signals.
    - **Machine Learning:** A simple Linear Regression model is trained on historical data to predict the next day's closing price based on the selected indicators.
    """)
