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
    print("💰 Generating Trading Signal...")
    
    try:
        df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # LOGIC:
    # If the Forecast Price is significantly LOWER than the average, we BUY (Long).
    # If the Forecast Price is significantly HIGHER, we SELL (Short).
    
    avg_price = df['Forecast'].mean()
    next_24h = df['Forecast'].iloc[:24].mean()
    
    signal = "NEUTRAL"
    if next_24h < avg_price * 0.9:
        signal = "LONG (BUY) 🟢"
        reason = "Prices are forecast to be 10% below average."
    elif next_24h > avg_price * 1.1:
        signal = "SHORT (SELL) 🔴"
        reason = "Prices are forecast to be 10% above average."
    else:
        reason = "Prices are stable."

    report = [
        "TRADING SIGNAL REPORT",
        "="*30,
        f"Signal: {signal}",
        f"Reason: {reason}",
        f"Next 24h Avg Price: €{next_24h:.2f}",
        f"Market Avg Price:   €{avg_price:.2f}",
        "="*30
    ]
    
    os.makedirs("outputs", exist_ok=True)
    with open(OUTPUT_SIGNAL, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
        
    print("\n".join(report))
    print(f"\n📄 Saved to {OUTPUT_SIGNAL}")

if __name__ == "__main__":
    generate_signal()