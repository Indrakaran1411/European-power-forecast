import pandas as pd
import matplotlib.pyplot as plt
import os

# CONFIGURATION
INPUT_FILE = "data/forecast_results.csv"
OUTPUT_PLOT = "outputs/forecast_plot.png"

def generate_plot():
    print("📊 Generating Forecast Plot...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"❌ Error reading {INPUT_FILE}: {e}")
        return

    # 2. Setup Plot
    # We plot the first week (168 hours) to show the "Shape" clearly to the recruiter.
    # Plotting a whole year looks messy.
    plt.figure(figsize=(14, 7))
    
    plot_data = df.iloc[:168]
    
    # Actual Price (Black line)
    plt.plot(plot_data.index, plot_data['Actual'], 
             label='Actual Market Price', color='black', linewidth=1.5, alpha=0.7)
    
    # AI Forecast (Green dashed)
    plt.plot(plot_data.index, plot_data['Forecast'], 
             label='AI Model Forecast', color='#2ca02c', linewidth=2, linestyle='--')
    
    # Styling
    plt.title("Netherlands Power Price Forecast (Best AI Model)", fontsize=14, fontweight='bold')
    plt.ylabel("Price (EUR/MWh)", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 3. Save
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"✅ Plot saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    generate_plot()