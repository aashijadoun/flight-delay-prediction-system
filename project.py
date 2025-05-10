import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

# ===== Set random seed for reproducibility =====
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# ===== Load data =====
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

# ===== Fit SARIMA =====
sarima_model = SARIMAX(daily_delay, order=(2, 1, 1), seasonal_order=(1, 1, 1, 7))
sarima_result = sarima_model.fit(disp=False)

# ===== Get residuals and normalize for LSTM =====
residuals = sarima_result.resid
scaler = MinMaxScaler()
res_scaled = scaler.fit_transform(residuals.values.reshape(-1, 1))

# ===== Create LSTM sequences =====
def create_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

window_size = 14
X_lstm, y_lstm = create_sequences(res_scaled, window_size)

# ===== Build and train LSTM model =====
model = Sequential([
    LSTM(64, activation='relu', input_shape=(window_size, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_lstm, y_lstm, epochs=50, batch_size=16, verbose=0,
          callbacks=[EarlyStopping(monitor='loss', patience=5)])

# ===== Forecast SARIMA for next n_days =====
sarima_forecast = sarima_result.forecast(steps=n_days)

# ===== Forecast residuals with LSTM =====
lstm_input = res_scaled[-window_size:].reshape(1, window_size, 1)
lstm_residuals = []

for _ in range(n_days):
    pred_res = model.predict(lstm_input, verbose=0)[0][0]
    lstm_residuals.append(pred_res)
    lstm_input = np.append(lstm_input[:, 1:, :], [[[pred_res]]], axis=1)

# ===== Inverse scale and combine forecasts =====
lstm_residuals = scaler.inverse_transform(np.array(lstm_residuals).reshape(-1, 1)).flatten()
hybrid_forecast = sarima_forecast + lstm_residuals

# ===== Plot results =====
last_date = daily_delay.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days)

plt.figure(figsize=(12, 6))
plt.plot(daily_delay, label='Historical Delay')
plt.plot(forecast_dates, hybrid_forecast, label='SARIMA + LSTM Forecast', color='red')
plt.title(f'Delay Forecast for Flight {flight_number} from {origin} to {destination}')
plt.xlabel('Date')
plt.ylabel('Avg Arrival Delay')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ===== Print forecast =====
print("\n===== Hybrid SARIMA + LSTM Forecasted Delays =====")
for date, value in zip(forecast_dates, hybrid_forecast):
    print(f"{date.strftime('%Y-%m-%d')} : {value:.2f} minutes")

# ===== Optional: Forecast performance evaluation (RMSE) =====
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

    print("\n📊 Forecast Performance on Last {} Days:".format(n_days))
    print("   RMSE : {:.2f} minutes".format(rmse))
