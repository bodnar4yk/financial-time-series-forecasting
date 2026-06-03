import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
df = pd.read_csv('sample.csv', index_col=0)

# Split into Train and Test samples (80 / 20)
split_idx = int(len(df) * 0.8)

train = df.iloc[:split_idx]
test = df.iloc[split_idx:]
len_test = len(test)

columns_list = ['Pay_Type_1', 'Pay_Type_2', 'Total_Payment']
results_list = []
future_forecasts_dict = {}  # Dictionary to save forecasts for all columns

for i in columns_list:
    # 1. Log-transform train data to stabilize variance
    train_log = np.log1p(train[i])

    # 2. Build and train SARIMAX(1, 1, 1)x(1, 1, 1, 7) model
    # disp=False suppresses iteration logs of the optimizer in the console
    model = SARIMAX(train_log, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    results = model.fit(disp=False)

    # 3. Generate forecast for the entire test set length
    forecast_log = results.forecast(steps=len_test)

    # 4. Inverse transformation back to the original scale
    forecast_original = np.expm1(forecast_log)

    # Save the first 7 steps of the forecast 
    # Since forecast_original is a pd.Series, we slice it using .iloc[:7]
    future_forecasts_dict[i] = forecast_original.iloc[:7].values

    # 5. Calculate performance metrics on the test set
    mae = mean_absolute_error(forecast_original, test[i])
    rmse = mean_squared_error(forecast_original, test[i])**0.5

    table = {
        'Model': 'SARIMAX',
        'set': i,
        'MAE': mae,
        'RMSE': rmse
    }
    results_list.append(table)

# Create and display the metrics DataFrame
df_metrics = pd.DataFrame(results_list)
print("--- Metrics ---")
print(df_metrics)
print("\n" + "="*40 + "\n")

# Create and display the final forecast DataFrame for the first 7 steps
forecast_index = [f"Step +{k}" for k in range(1, 8)]
df_future = pd.DataFrame(future_forecasts_dict, index=forecast_index)

print("--- Forecast for the first 7 steps of test set ---")
print(df_future)