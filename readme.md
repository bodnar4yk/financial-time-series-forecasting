# Financial Time Series Forecasting Benchmark

This repository focuses on building and benchmarking various time-series forecasting approaches to predict financial metrics (`Pay_Type_1`, `Pay_Type_2`, and `Total_Payment`). 

Due to a high volume of missing values at the daily level, the raw data underwent a resampling and data-cleaning phase. The final dataset was aggregated on a **monthly basis** to maximize data density, stabilize variance, and capture robust mid-to-long-term macroeconomic trends.

## 📁 Repository Structure
* `sample.csv` — The cleaned and preprocessed dataset, aggregated into monthly intervals.
* `forecast_Arima.py` — Classical statistical modeling using the ARIMA framework.
* `forecast_sarimax.py` — Seasonal autoregressive integrated moving average modeling (capturing monthly patterns).
* `forecast_XGboost_RandomForest.py` — Machine learning approach utilizing Tree-based models with engineered lag and rolling features.
* `forecast_prophet.py` — Meta's Prophet model optimized for business-cycle trends.

## 🛠️ Technological Stack
* **Language:** Python 3.x
* **Core Libraries:** Pandas, NumPy, Scikit-learn
* **Forecasting Frameworks:** Statsmodels (ARIMA/SARIMAX), XGBRegressor, RandomForestRegressor, Prophet

## 📊 Feature Engineering & Validation Strategy
For the machine learning models (XGBoost & Random Forest), specific features were engineered to provide temporal context to the tree split decisions:
* **Lag Features:** 7-period historical shifts (`lag_1` to `lag_7`) to maintain autoregressive context.
* **Rolling Windows:** 3-period and 7-period rolling means and standard deviations to capture moving trends and volatility shifts without causing data leakage.
* **Validation:** A sequential time-series split was implemented (80% Train / 20% Test) to strictly preserve temporal order and evaluate true out-of-sample forecasting performance.

## 📉 Evaluation & Insights (Benchmark Results)
Models were evaluated on the test set using Mean Absolute Error (**MAE**) and Root Mean Squared Error (**RMSE**). 

* **Tree-based ML (XGBoost/Random Forest):** Demonstrated superior performance in minimizing RMSE on volatile payment streams. The introduction of shifting rolling windows allowed the trees to react to trend shifts far better than baseline statistical models.
* **ARIMA/SARIMAX:** Performed decently on steady components but tended to mean-revert or damp out into a flat constant line too rapidly over longer horizons.
* **Meta Prophet:** Highly sensitive to structural trend breaks in compact datasets. While stable on linear paths, it required strict constraint tuning (`changepoint_prior_scale`) to prevent overshooting under extreme volatility.

## 🚀 How to Run the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/bodnar4yk/financial-time-series-forecasting.git

2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost statsmodels prophet
   ```

3. Run any specific model script, for example:
   ```bash
   python forecast_XGboost_RandomForest.py
