# 🇳🇱 Dutch Power Price Forecast AI

An End-to-End MLOps pipeline to forecast Day-Ahead electricity prices in the Netherlands.

## 🏆 Project Results

### 1. Model Performance
After rigorously comparing Ridge Regression, XGBoost, and a Structural Hybrid model, **XGBoost** was selected as the winner.
* **MAE:** €30.36
* **RMSE:** €42.15
* **R²:** 0.45

### 2. Visual Forecast
![Forecast Plot](outputs/forecast_plot.png)

### 3. Business Value (Trading Signal)
*Generated automatically by `src/trading_signal.py`*
> **Signal:** LONG (BUY) 🟢  
> **Reason:** Prices are forecast to be 10% below average.  
> **Next 24h Avg:** €27.38  

---

## 🛠️ Tech Stack
* **Modeling:** XGBoost, Scikit-Learn
* **Experiment Tracking:** MLflow
* **GenAI:** Google Gemini (Automated Reporting)
* **Engineering:** Pandas, NumPy

## 🚀 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the pipeline: `python src/modeling.py`