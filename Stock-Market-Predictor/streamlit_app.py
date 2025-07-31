
# Load and rename columns if necessary
import pandas as pd




import streamlit as st
import pandas as pd

st.title("📈 Stock Market Predictor")

uploaded_file = st.file_uploader("Upload your ADANIPORTS CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    st.success("✅ File uploaded successfully!")
    
    # Show the dataframe
    st.write("### Preview of Uploaded Data", df.head())

    # Parse date column if available
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        st.line_chart(df.set_index("Date")["Close"])
    else:
        st.error("❌ 'Date' column not found.")
else:
    st.warning("⚠️ Please upload a CSV file to continue.")

  # example
import os
st.write("Visible files:", os.listdir())


# Clean up column names (remove spaces, weird chars)
df.columns = df.columns.str.strip()

# Debug: show column names
import streamlit as st
st.write("✅ Columns in CSV:", df.columns.tolist())

# Now safely parse the 'Date' column
df['Date'] = pd.to_datetime(df['Date'])
import os
import streamlit as st
st.write("Current working directory:", os.getcwd())
st.write("Files in directory:", os.listdir())






import streamlit as st
import pandas as pd

uploaded_file = st.file_uploader("Upload your CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

# Rename to standard column names
df.rename(columns={
    'date': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Convert date and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Drop NaNs if any
df.dropna(inplace=True)

# streamlit_app.py



import pandas as pd
import ta
import mplfinance as mpf
# ... rest of your code

import ta
import mplfinance as mpf

# VWAP
vwap = ta.volume.VolumeWeightedAveragePrice(
    high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume']
)
data['VWAP'] = vwap.volume_weighted_average_price()

# EMAs
data['EMA20'] = ta.trend.EMAIndicator(close=data['Close'], window=20).ema_indicator()
data['EMA50'] = ta.trend.EMAIndicator(close=data['Close'], window=50).ema_indicator()

# Bollinger Bands
bb = ta.volatility.BollingerBands(close=data['Close'], window=20, window_dev=2)
data['BB_upper'] = bb.bollinger_hband()
data['BB_lower'] = bb.bollinger_lband()

add_plots = [
    mpf.make_addplot(data['EMA20'], color='blue', width=1.5),
    mpf.make_addplot(data['EMA50'], color='orange', width=1.5),
    mpf.make_addplot(data['VWAP'], color='purple', linestyle=':', width=1.5),
    mpf.make_addplot(data['BB_upper'], color='gray', linestyle='--', width=1),
    mpf.make_addplot(data['BB_lower'], color='gray', linestyle='--', width=1)
]

# Plot and save the high-quality chart
mpf.plot(
    data,
    type='candle',
    volume=True,
    style='yahoo',
    title="ADANIPORTS Strategy Chart (VWAP, EMA, Bollinger Bands)",
    addplot=add_plots,
    figscale=2.0,           # Increase scale for clarity
    figratio=(18, 10),      # Wide aspect ratio
    savefig='figure1_strategy_chart.png'
)

from google.colab import files
files.download('figure1_strategy_chart.png')

# Zoom into a specific range, e.g., a range present in the data
zoom_data = data.loc['2021-01-01':'2021-01-31']

add_plots_zoom = [
    mpf.make_addplot(zoom_data['EMA20'], color='blue'),
    mpf.make_addplot(zoom_data['EMA50'], color='orange'),
    mpf.make_addplot(zoom_data['VWAP'], color='purple', linestyle=':'),
    mpf.make_addplot(zoom_data['BB_upper'], color='gray', linestyle='--'),
    mpf.make_addplot(zoom_data['BB_lower'], color='gray', linestyle='--')
]

mpf.plot(
    zoom_data,
    type='candle',
    volume=True,
    title="Zoomed View - VWAP, EMA, BB (Jan 2023)",
    style='yahoo',
    addplot=add_plots_zoom,
    figscale=2.2,
    figratio=(16, 9),
    savefig='figure_zoomed_section.png'
)

from google.colab import files
files.download('figure_zoomed_section.png')

# Buy condition: EMA20 > EMA50 and Close > VWAP
buy_signals = (data['EMA20'] > data['EMA50']) & (data['Close'] > data['VWAP'])

# Sell condition: EMA20 < EMA50 and Close < VWAP
sell_signals = (data['EMA20'] < data['EMA50']) & (data['Close'] < data['VWAP'])

# Create buy/sell markers
data['Buy'] = data['Low'][buy_signals] * 0.98  # slightly below candle for arrow
data['Sell'] = data['High'][sell_signals] * 1.02  # slightly above candle

import mplfinance as mpf

apds = [
    mpf.make_addplot(data['EMA20'], color='blue'),
    mpf.make_addplot(data['EMA50'], color='orange'),
    mpf.make_addplot(data['VWAP'], color='purple', linestyle=':'),
    mpf.make_addplot(data['BB_upper'], color='gray', linestyle='--'),
    mpf.make_addplot(data['BB_lower'], color='gray', linestyle='--'),

    # Arrows
    mpf.make_addplot(data['Buy'], type='scatter', marker='^', markersize=200,
                     color='green', panel=0),
    mpf.make_addplot(data['Sell'], type='scatter', marker='v', markersize=200,
                     color='red', panel=0)
]

mpf.plot(
    data,
    type='candle',
    volume=True,
    style='yahoo',
    title='Buy/Sell Signal Chart with VWAP, EMA, Bollinger Bands',
    addplot=apds,
    figscale=2.2,
    figratio=(16, 9),
    savefig='figure2_buy_sell_chart.png'
)

from google.colab import files
files.download('figure2_buy_sell_chart.png')

# Choose a short time window to zoom in
zoom_data = data.loc['2021-01-01':'2021-01-31'].copy()

# Recalculate conditions in zoomed data
zoom_data['Buy'] = zoom_data['Low'][
    (zoom_data['EMA20'] > zoom_data['EMA50']) & (zoom_data['Close'] > zoom_data['VWAP'])
] * 0.98

zoom_data['Sell'] = zoom_data['High'][
    (zoom_data['EMA20'] < zoom_data['EMA50']) & (zoom_data['Close'] < zoom_data['VWAP'])
] * 1.02

apds_zoom = [
    mpf.make_addplot(zoom_data['EMA20'], color='blue'),
    mpf.make_addplot(zoom_data['EMA50'], color='orange'),
    mpf.make_addplot(zoom_data['VWAP'], color='purple', linestyle=':'),
    mpf.make_addplot(zoom_data['BB_upper'], color='gray', linestyle='--'),
    mpf.make_addplot(zoom_data['BB_lower'], color='gray', linestyle='--'),

    # Buy/Sell markers
    mpf.make_addplot(zoom_data['Buy'], type='scatter', marker='^', markersize=200,
                     color='green', panel=0),
    mpf.make_addplot(zoom_data['Sell'], type='scatter', marker='v', markersize=200,
                     color='red', panel=0)
]

mpf.plot(
    zoom_data,
    type='candle',
    volume=True,
    style='yahoo',
    title='Zoomed Buy/Sell Chart (Jan 2023) with Indicators',
    addplot=apds_zoom,
    figscale=2.2,
    figratio=(18, 10),
    savefig='figure_zoomed_buy_sell.png'
)

