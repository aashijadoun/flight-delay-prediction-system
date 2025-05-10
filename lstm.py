import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

# ===== Reproducibility =====
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# ===== Load and preprocess data =====
df = pd.read_csv('flights.csv', low_memory=False)
df.columns = df.columns.str.strip()
df['FL_DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])

# ===== User input =====
flight_number = input("Enter Flight Number: ")
origin = input("Enter Origin Airport: ")
destination = input("Enter Destination Airport: ")
n_days = int(input("Enter number of days to forecast: "))

# ===== Filter data =====
filtered = df[
    (df['FLIGHT_NUMBER'].astype(str) == flight_number) &
    (df['ORIGIN_AIRPORT'] == origin) &
    (df['DESTINATION_AIRPORT'] == destination)
]

if filtered.empty:
    print("❌ No flight data found for the specified inputs. Please try different values.")
    exit()

# ===== Aggregate delay by date =====
daily_delay = filtered.groupby('FL_DATE')['ARRIVAL_DELAY'].mean().fillna(0)

# ===== Normalize =====
scaler = MinMaxScaler()
scaled_delay = scaler.fit_transform(daily_delay.values.reshape(-1, 1))

# ===== Create sequences =====
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

window_size = 14
X, y = create_sequences(scaled_delay, window_size)

# ===== LSTM Model =====
model = Sequential([
    LSTM(64, activation='relu', input_shape=(window_size, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, batch_size=16, verbose=0,
          callbacks=[EarlyStopping(monitor='loss', patience=5)])

# ===== Forecast future values =====
lstm_input = scaled_delay[-window_size:].reshape(1, window_size, 1)
forecast = []

for _ in range(n_days):
    pred = model.predict(lstm_input, verbose=0)[0][0]
    forecast.append(pred)
    lstm_input = np.append(lstm_input[:, 1:, :], [[[pred]]], axis=1)

forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()

# ===== Plot results =====
last_date = daily_delay.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days)

plt.figure(figsize=(12, 6))
plt.plot(daily_delay.index, daily_delay.values, label='Historical Delay', color='blue')
plt.plot(forecast_dates, forecast, label='LSTM Forecast', color='orange')
plt.title(f'LSTM Forecast: Flight {flight_number} ({origin} ➝ {destination})')
plt.xlabel('Date')
plt.ylabel('Avg Arrival Delay (minutes)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ===== Print forecast =====
print("\n===== LSTM Forecasted Delays =====")
for date, val in zip(forecast_dates, forecast):
    print(f"{date.strftime('%Y-%m-%d')} : {val:.2f} minutes")

# ===== RMSE Evaluation on Last n_days =====
if len(scaled_delay) > window_size + n_days:
    # Split into train and test
    train_scaled = scaled_delay[:-n_days]
    test_actual = daily_delay[-n_days:].values

    # Create training sequences
    X_train, y_train = create_sequences(train_scaled, window_size)

    # Retrain model on training data
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0,
              callbacks=[EarlyStopping(monitor='loss', patience=5)])

    # Prepare input for forecasting last n_days
    lstm_input = train_scaled[-window_size:].reshape(1, window_size, 1)
    test_forecast_scaled = []

    for _ in range(n_days):
        pred = model.predict(lstm_input, verbose=0)[0][0]
        test_forecast_scaled.append(pred)
        lstm_input = np.append(lstm_input[:, 1:, :], [[[pred]]], axis=1)

    test_forecast = scaler.inverse_transform(np.array(test_forecast_scaled).reshape(-1, 1)).flatten()

    # Calculate RMSE
    rmse_lstm = np.sqrt(mean_squared_error(test_actual, test_forecast))
    print(f"\n📊 LSTM RMSE on last {n_days} days: {rmse_lstm:.2f} minutes")
else:
    print("\n⚠️ Not enough data to compute RMSE for LSTM.")
