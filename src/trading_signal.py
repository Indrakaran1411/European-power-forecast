import pandas as pd
import numpy as np
import os
import sys

# Windows Emoji Fix
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

INPUT_FILE = "data/forecast_results.csv"
OUTPUT_SIGNAL = "outputs/trading_view.txt"

def generate_signal():
    print("💰 Generating Trading Signal (Rolling 7-Day Strategy)...")
    
    try:
        df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # LOGIC UPDATE: Compare "Tomorrow" vs "Last 7 Days"
    # We simulate this by taking the LAST 24 hours of our data as "Tomorrow"
    # and the 168 hours (7 days) BEFORE that as the "Baseline".
    
    # 1. Select the relevant windows
    tomorrow_view = df.iloc[-24:] # The final day in our dataset
    history_view = df.iloc[-192:-24] # The 7 days prior to that final day
    
    if len(history_view) < 168:
        print("⚠️ Not enough data for 7-day rolling comparison. Using available history.")
    
    # 2. Calculate Metrics
    forecast_price = tomorrow_view['Forecast'].mean()
    baseline_price = history_view['Actual'].mean() # Compare against what actually happened recently
    
    # 3. Define Thresholds (e.g., +/- 5% deviation)
    threshold = 0.05 
    
    signal = "NEUTRAL ⚪"
    reason = "Price is within normal 7-day range."
    
    if forecast_price < baseline_price * (1 - threshold):
        signal = "LONG (BUY) 🟢"
        diff = (1 - forecast_price/baseline_price) * 100
        reason = f"Forecast is {diff:.1f}% CHEAPER than the 7-day average."
    elif forecast_price > baseline_price * (1 + threshold):
        signal = "SHORT (SELL) 🔴"
        diff = (forecast_price/baseline_price - 1) * 100
        reason = f"Forecast is {diff:.1f}% MORE EXPENSIVE than the 7-day average."

    # 4. Generate Report
    target_date = tomorrow_view.index[0].strftime('%Y-%m-%d')
    
    report = [
        "TRADING SIGNAL REPORT",
        "="*40,
        f"Target Date:       {target_date}",
        f"Strategy:          Mean Reversion (vs 7-Day Rolling Avg)",
        "-"*40,
        f"Signal:            {signal}",
        f"Reason:            {reason}",
        "-"*40,
        f"Forecast Price:    €{forecast_price:.2f}",
        f"7-Day Baseline:    €{baseline_price:.2f}",
        "="*40,
        "\nInvalidation Logic:",
        "Discard signal if real-time Interconnector capacity drops >500MW",
        "after 10:00 UTC, as this changes the fundamental supply curve."
    ]
    
    os.makedirs("outputs", exist_ok=True)
    with open(OUTPUT_SIGNAL, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
        
    print("\n".join(report))
    print(f"\n📄 Saved to {OUTPUT_SIGNAL}")

if __name__ == "__main__":
    generate_signal()