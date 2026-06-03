import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
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

for i in columns_list:
    # 1. Log train sample to stabilize variance
    train_log = np.log1p(train[i])

    # 2. Build and train ARIMA(1, 1, 1) model
    model = ARIMA(train_log, order=(1, 1, 1))
    results = model.fit()

    # 3. Forecast for log values
    forecast_log = results.forecast(steps=len_test)

    # 4. Inverse transformation back to original data scale
    forecast_original = np.expm1(forecast_log)

    # 5. Calculate metrics on the test set
    mae = mean_absolute_error(forecast_original, test[i])
    rmse = mean_squared_error(forecast_original, test[i])**0.5

    table = {
        'Model': 'ARIMA',
        'set': i,
        'MAE': mae,
        'RMSE': rmse
    }
    results_list.append(table)

# Create and display metrics DataFrame
df_metrics = pd.DataFrame(results_list)

print("--- Performance Metrics ---")
print(df_metrics)

# Display only the first 7 forecast steps of the test set
print("\n--- Forecast for the first 7 test steps ---")
print(forecast_original.head(7))