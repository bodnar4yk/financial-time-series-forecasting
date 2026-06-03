import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load data
df = pd.read_csv('sample.csv', index_col=0)

columns_list = ['Pay_Type_1', 'Pay_Type_2', 'Total_Payment']
results_list = []
future_forecasts_dict = {}

# Calculate the split index for sequential train-test split (80 / 20)
split_idx = int(len(df) * 0.8)

# Create a clean synthetic date range
# This ensures Prophet has an evenly spaced daily timeline to stabilize trend estimation
clean_dates = pd.date_range(start='2000-01-01', periods=len(df), freq='D')

for i in columns_list:
    # Prepare the dataframe strictly adhering to Prophet's expected input format ('ds' and 'y')
    df_prophet = pd.DataFrame({
        'ds': clean_dates,
        'y': df[i].values
    })
    
    # Split data into sequential Train and Test sets
    train = df_prophet.iloc[:split_idx]
    test = df_prophet.iloc[split_idx:]
    len_test = len(test)
    
    # 2. Configure Prophet model
    # Yearly seasonality is disabled since the synthetic range might distort calendar years
    # changepoint_prior_scale limits trend flexibility to prevent mathematical instability/explosion
    model = Prophet(
        yearly_seasonality=False, 
        weekly_seasonality=True, 
        daily_seasonality=False,
        changepoint_prior_scale=0.05 
    )
    
    # Fit the model (Prophet automatically suppresses logs if verbose is not configured)
    model.fit(train)
    
    # 3. Generate forecast
    # Construct a future dataframe for the duration of the test set length
    future_dates = model.make_future_dataframe(periods=len_test, freq='D')
    forecast = model.predict(future_dates)
    
    # Extract the predicted values ('yhat') specifically matching the test period window
    test_predictions = forecast['yhat'].iloc[split_idx:].values
    
    # Store the first 7 steps of the forecast for final comparison
    future_forecasts_dict[i] = test_predictions[:7]
    
    # 4. Calculate performance metrics on the test set
    mae = mean_absolute_error(test_predictions, test['y'])
    rmse = mean_squared_error(test_predictions, test['y'])**0.5
    
    table = {
        'Model': 'Prophet',
        'set': i,
        'MAE': mae,
        'RMSE': rmse
    }
    results_list.append(table)

# Create and display performance metrics DataFrame
df_metrics = pd.DataFrame(results_list)
print("--- Fixed Prophet Metrics ---")
print(df_metrics)
print("\n" + "="*50 + "\n")

# Create and display the final forecast DataFrame for the first 7 test steps
forecast_index = [f"Step +{k}" for k in range(1, 8)]
df_future = pd.DataFrame(future_forecasts_dict, index=forecast_index)
print("--- Fixed Prophet Forecast (First 7 steps) ---")
print(df_future)