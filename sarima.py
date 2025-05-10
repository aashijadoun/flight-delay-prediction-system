import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# ===== Set random seed for reproducibility =====
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)

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

# ===== Forecast SARIMA for next n_days =====
sarima_forecast = sarima_result.forecast(steps=n_days)
forecast_dates = pd.date_range(start=daily_delay.index[-1] + pd.Timedelta(days=1), periods=n_days)

# ===== Plot SARIMA Forecast =====
plt.figure(figsize=(12, 6))
plt.plot(daily_delay, label='Historical Delay')
plt.plot(forecast_dates, sarima_forecast, label='SARIMA Forecast', color='blue')
plt.title(f'Delay Forecast for Flight {flight_number} from {origin} to {destination} (SARIMA Only)')
plt.xlabel('Date')
plt.ylabel('Avg Arrival Delay')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ===== Print SARIMA Forecast =====
print("\n===== SARIMA Forecasted Delays =====")
for date, value in zip(forecast_dates, sarima_forecast):
    print(f"{date.strftime('%Y-%m-%d')} : {value:.2f} minutes")

# ===== Evaluate RMSE on last n_days if enough data =====
if len(daily_delay) >= n_days:
    train_data = daily_delay[:-n_days]
    test_data = daily_delay[-n_days:]

    eval_model = SARIMAX(train_data, order=(2, 1, 1), seasonal_order=(1, 1, 1, 7))
    eval_result = eval_model.fit(disp=False)
    eval_forecast = eval_result.forecast(steps=n_days)

    rmse = np.sqrt(mean_squared_error(test_data, eval_forecast))
    print(f"\n📊 SARIMA RMSE on last {n_days} days: {rmse:.2f} minutes")
else:
    print("\n⚠️ Not enough historical data to compute RMSE.")

