import os
import random
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

# ----- PAGE CONFIG -----
st.set_page_config(page_title="Flight Delay Forecast", layout="wide")

# ----- BACKGROUND SETUP -----
def add_bg_from_image(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
        b64_img = base64.b64encode(data).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{b64_img}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color: #FFFFFF;
        }}
        .css-1d391kg {{
            background-color: rgba(0,0,0,0.6) !important;
        }}
        .block-container {{
            background-color: rgba(0, 0, 0, 0.5);
            padding: 2rem;
            border-radius: 1rem;
        }}
        </style>
    """, unsafe_allow_html=True)

add_bg_from_image("254381.jpg")

# ----- RANDOM SEED -----
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# ----- LOAD DATA -----
@st.cache_data
def load_data():
    df = pd.read_csv('flights.csv', low_memory=False)
    df.columns = df.columns.str.strip()
    df['FL_DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
    return df

df = load_data()

# ----- TITLE -----
st.title("✈️ Flight Delay Forecast with Hybrid SARIMA + LSTM")

# ----- SIDEBAR INPUTS -----
st.sidebar.header("Enter Flight Details")

if 'flight_num' not in st.session_state:
    st.session_state.flight_num = "1227"
if 'origin' not in st.session_state:
    st.session_state.origin = "SFO"
if 'dest' not in st.session_state:
    st.session_state.dest = "CLT"
if 'forecast_len' not in st.session_state:
    st.session_state.forecast_len = 7

st.sidebar.text_input("Flight Number (e.g., 1227):", key="flight_num")
st.sidebar.text_input("Origin Airport (e.g., SFO):", key="origin")
st.sidebar.text_input("Destination Airport (e.g., CLT):", key="dest")
st.sidebar.number_input("Forecast Days Ahead:", min_value=1, max_value=30, key="forecast_len")

run_forecast = st.sidebar.button("🚀 Run Forecast")

# ----- MAIN FORECAST LOGIC -----
if run_forecast:
    flight_number = st.session_state.flight_num
    origin = st.session_state.origin.upper()
    destination = st.session_state.dest.upper()
    n_days = st.session_state.forecast_len

    filtered = df[
        (df['FLIGHT_NUMBER'].astype(str) == flight_number) &
        (df['ORIGIN_AIRPORT'] == origin) &
        (df['DESTINATION_AIRPORT'] == destination)
    ]

    if filtered.empty:
        st.error("❌ No flight data found for the specified inputs. Please try different values.")
    else:
        daily_delay = filtered.groupby('FL_DATE')['ARRIVAL_DELAY'].mean().fillna(0)

        # SARIMA Model
        sarima_model = SARIMAX(daily_delay, order=(2, 1, 1), seasonal_order=(1, 1, 1, 7))
        sarima_result = sarima_model.fit(disp=False)

        # Residuals for LSTM
        residuals = sarima_result.resid
        scaler = MinMaxScaler()
        res_scaled = scaler.fit_transform(residuals.values.reshape(-1, 1))

        def create_sequences(data, window):
            X, y = [], []
            for i in range(window, len(data)):
                X.append(data[i - window:i])
                y.append(data[i])
            return np.array(X), np.array(y)

        window_size = 14
        X_lstm, y_lstm = create_sequences(res_scaled, window_size)

        # LSTM Model
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(window_size, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_lstm, y_lstm, epochs=50, batch_size=16, verbose=0,
                  callbacks=[EarlyStopping(monitor='loss', patience=5)])

        # Forecast SARIMA
        sarima_forecast = sarima_result.forecast(steps=n_days)

        # Forecast Residuals with LSTM
        lstm_input = res_scaled[-window_size:].reshape(1, window_size, 1)
        lstm_residuals = []

        for _ in range(n_days):
            pred_res = model.predict(lstm_input, verbose=0)[0][0]
            lstm_residuals.append(pred_res)
            lstm_input = np.append(lstm_input[:, 1:, :], [[[pred_res]]], axis=1)

        lstm_residuals = scaler.inverse_transform(np.array(lstm_residuals).reshape(-1, 1)).flatten()
        hybrid_forecast = sarima_forecast + lstm_residuals

        # Plot Forecast
        last_date = daily_delay.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(daily_delay, label='Historical Delay')
        ax.plot(forecast_dates, hybrid_forecast, label='SARIMA + LSTM Forecast', color='red')
        ax.set_title(f'Delay Forecast for Flight {flight_number} from {origin} to {destination}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Avg Arrival Delay')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Show Forecast
        st.subheader("📅 Forecasted Delays")
        for date, value in zip(forecast_dates, hybrid_forecast):
            st.write(f"{date.strftime('%Y-%m-%d')} : {value:.2f} minutes")

        # Optional RMSE Evaluation
        if len(daily_delay) >= n_days:
            true_values = daily_delay[-n_days:].values
            past_sarima = sarima_result.get_forecast(steps=n_days)
            past_sarima_forecast = past_sarima.predicted_mean.values
            past_residuals = residuals[-n_days:].values

            past_res_scaled = scaler.transform(past_residuals.reshape(-1, 1))
            X_test, _ = create_sequences(np.concatenate((res_scaled[:-n_days], past_res_scaled)), window_size)
            lstm_preds = model.predict(X_test[-n_days:], verbose=0).flatten()
            lstm_preds = scaler.inverse_transform(lstm_preds.reshape(-1, 1)).flatten()

            hybrid_eval_forecast = past_sarima_forecast + lstm_preds
            rmse = np.sqrt(mean_squared_error(true_values, hybrid_eval_forecast))

            st.subheader(f"📊 Forecast Performance on Last {n_days} Days")
            st.write(f"RMSE: {rmse:.2f} minutes")
else:
    st.info("Enter flight details and click 'Run Forecast' to begin.")
