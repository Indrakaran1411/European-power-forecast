import sys
import os

# --- FIX: FORCE WINDOWS TO HANDLE EMOJIS ---
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass 
# -------------------------------------------

import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost

# Metric Imports
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin

# =========================
# 1. CONFIGURATION
# =========================
INPUT_FILE = "data/submission_data.csv"
OUTPUT_FILE = "data/forecast_results.csv"
MLFLOW_EXP_NAME = "Power_Price_Forecast_NL_1"

# =========================
# 2. FEATURE ENGINEERING
# =========================
def feature_eng(df):
    df = df.copy()
    # Time
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    
    # Physics (Residual Load = Demand - Renewables)
    df['Residual_Load'] = df['Load_MW'] - (df['Solar'] + df['Wind_Onshore'] + df['Wind_Offshore'])
    
    # Lags
    df['Price_Lag_24'] = df['Price_EUR'].shift(24)
    df['Price_Lag_168'] = df['Price_EUR'].shift(168)
    
    return df.dropna()

# =========================
# 3. HYBRID MODEL CLASS
# =========================
class SimpleHybrid(BaseEstimator, RegressorMixin):
    def __init__(self, linear_model=None, xgb_model=None):
        self.linear_model = linear_model if linear_model else Ridge()
        self.xgb_model = xgb_model if xgb_model else xgb.XGBRegressor()
        
    def fit(self, X, y):
        self.linear_model.fit(X, y)
        linear_preds = self.linear_model.predict(X)
        residuals = y - linear_preds
        self.xgb_model.fit(X, residuals)
        return self

    def predict(self, X):
        return self.linear_model.predict(X) + self.xgb_model.predict(X)

# =========================
# 4. EXPERIMENTATION ENGINE
# =========================
def evaluate_model(y_true, y_pred, model_name):
    """Calculates and prints all metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"   📊 {model_name} Metrics:")
    print(f"      MAE:  {mae:.2f} EUR")
    print(f"      RMSE: {rmse:.2f} EUR")
    print(f"      R2:   {r2:.3f}")
    
    return mae, rmse, r2

def run_experiments():
    print("⚙️ Loading Data...")
    try:
        df = pd.read_csv(INPUT_FILE, parse_dates=['Timestamp_UTC'], index_col='Timestamp_UTC')
    except Exception as e:
        print(f"❌ Error reading data: {e}")
        return

    df = feature_eng(df)
    
    # Split
    split_date = "2023-11-01"
    train = df.loc[:split_date]
    test = df.loc[split_date:]
    test = test.iloc[24:] 

    X_train = train.drop(columns=['Price_EUR'])
    y_train = train['Price_EUR']
    X_test = test.drop(columns=['Price_EUR'])
    y_test = test['Price_EUR']

    mlflow.set_experiment(MLFLOW_EXP_NAME)
    print(f"🧪 Starting MLflow Experiment: {MLFLOW_EXP_NAME}")

    # Track Best Model
    best_mae = float('inf')
    best_model_name = "None"
    best_preds = None

    # --- 1. BASELINE ---
    print("\n--- 1. Baseline (Seasonal Naive) ---")
    base_preds = X_test['Price_Lag_168']
    mae, rmse, r2 = evaluate_model(y_test, base_preds, "Baseline")
    
    if mae < best_mae:
        best_mae = mae
        best_model_name = "Baseline"
        best_preds = base_preds

    # --- 2. RIDGE REGRESSION ---
    with mlflow.start_run(run_name="Ridge_Regression"):
        print("\n--- 2. Regularized Regression (Ridge) ---")
        pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge())])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        
        mae, rmse, r2 = evaluate_model(y_test, preds, "Ridge")
        
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        if mae < best_mae:
            best_mae = mae
            best_model_name = "Ridge_Regression"
            best_preds = preds

    # --- 3. XGBOOST ---
    with mlflow.start_run(run_name="XGBoost_Tuned"):
        print("\n--- 3. XGBoost (RandomizedSearchCV) ---")
        xgb_model = xgb.XGBRegressor(n_jobs=-1, random_state=42)
        tscv = TimeSeriesSplit(n_splits=3)
        # Fast Grid Search
        param_dist = {'n_estimators': [500], 'learning_rate': [0.05], 'max_depth': [5]} 
        
        search = RandomizedSearchCV(xgb_model, param_dist, n_iter=1, cv=tscv, scoring='neg_mean_absolute_error')
        search.fit(X_train, y_train)
        preds = search.predict(X_test)
        
        mae, rmse, r2 = evaluate_model(y_test, preds, "XGBoost")
        
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        if mae < best_mae:
            best_mae = mae
            best_model_name = "XGBoost"
            best_preds = preds

    # --- 4. STRUCTURAL HYBRID ---
    with mlflow.start_run(run_name="Structural_Hybrid"):
        print("\n--- 4. Structural Hybrid (Ridge + XGB) ---")
        hybrid = SimpleHybrid(
            linear_model=Ridge(),
            xgb_model=xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5)
        )
        hybrid.fit(X_train, y_train)
        preds = hybrid.predict(X_test)
        
        mae, rmse, r2 = evaluate_model(y_test, preds, "Hybrid")
        
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        if mae < best_mae:
            best_mae = mae
            best_model_name = "Structural_Hybrid"
            best_preds = preds

    # =========================
    # 5. SAVE WINNER
    # =========================
    print(f"\n🏆 The Winner is: {best_model_name} (MAE: {best_mae:.2f})")
    print(f"💾 Saving forecast to {OUTPUT_FILE}...")
    
    results = pd.DataFrame({
        'Actual': y_test, 
        'Forecast': best_preds, 
        'Residual_Load': X_test['Residual_Load']
    })
    results.to_csv(OUTPUT_FILE)
    print("✅ Done.")

if __name__ == "__main__":
    run_experiments()