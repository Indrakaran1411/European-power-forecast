import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import sys

# Windows Emoji Fix
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
INPUT_FILE = "data/forecast_results.csv"
OUTPUT_LOG = "outputs/ai_logs.txt"

def generate_report():
    print("🤖 Generating AI Executive Summary...")
    
    if not API_KEY:
        print("❌ Error: GEMINI_API_KEY not found in .env")
        return

    try:
        df = pd.read_csv(INPUT_FILE)
        # Summarize data for the prompt (don't send whole file)
        avg_price = df['Forecast'].mean()
        max_price = df['Forecast'].max()
        min_price = df['Forecast'].min()
        mae_approx = (df['Actual'] - df['Forecast']).abs().mean()
    except Exception as e:
        print(f"❌ Error reading data: {e}")
        return

    prompt = f"""
    You are a Senior Power Trader. Write a 3-sentence executive summary about the Dutch Power Market based on this AI Forecast model:
    - Average Forecast Price: {avg_price:.2f} EUR
    - Max Price: {max_price:.2f} EUR
    - Min Price: {min_price:.2f} EUR
    - Model MAE Error: {mae_approx:.2f} EUR
    
    Explain if the market is volatile and if the model is trustworthy based on the MAE.
    """

    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        
        report_text = f"🗣️ GEMINI EXECUTIVE SUMMARY:\n{response.text}"
        
        with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
            f.write(report_text)
            
        print(report_text)
        print(f"\n📄 Saved to {OUTPUT_LOG}")
        
    except Exception as e:
        print(f"❌ AI Error: {e}")

if __name__ == "__main__":
    generate_report()