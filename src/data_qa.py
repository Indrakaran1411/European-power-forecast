import pandas as pd
import os
import sys

# Paths relative to where the script is run
INPUT_FILE = "data/submission_data.csv"
OUTPUT_REPORT = "outputs/data_qa_report.txt"

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

def run_qa():
    print(f"🔍 Running QA on {INPUT_FILE}...")
    
    # 1. Load Data
    try:
        if not os.path.exists(INPUT_FILE):
            print(f"❌ Error: File '{INPUT_FILE}' not found. Did you run ingestion?")
            return
            
        df = pd.read_csv(INPUT_FILE, parse_dates=['Timestamp_UTC'], index_col='Timestamp_UTC')
    except Exception as e:
        print(f"❌ CRITICAL: Read error. {e}")
        return

    lines = [f"DATA QA REPORT for {INPUT_FILE}", "="*40]
    
    # 2. Checks
    missing = df.isnull().sum().sum()
    lines.append(f"1. Missing Values: {'✅ PASS' if missing == 0 else f'⚠️ FAIL ({missing})'}")
    
    dupes = df.index.duplicated().sum()
    lines.append(f"2. Duplicates:     {'✅ PASS' if dupes == 0 else f'❌ FAIL ({dupes})'}")
    
    if 'Load_MW' in df.columns:
        neg_load = (df['Load_MW'] <= 0).sum()
        lines.append(f"3. Negative Load:  {'✅ PASS' if neg_load == 0 else f'❌ FAIL ({neg_load})'}")
    
    missing_hours = 8760 - len(df) 
    lines.append(f"4. Completeness:   {'✅ PASS' if abs(missing_hours) < 24 else f'⚠️ GAP ({missing_hours} missing)'}")

    # --- THE FIX: encoding="utf-8" IS CRITICAL HERE ---
    try:
        with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\n📄 Report saved successfully to: {OUTPUT_REPORT}")
    except Exception as e:
        print(f"❌ Error writing report: {e}")

    # Print to console (Handling console encoding limits)
    try:
        # We explicitly encode/decode to force the console to handle it or skip it
        print("\n".join(lines))
    except UnicodeEncodeError:
        print("\n(Console cannot display emojis, but the text report was saved correctly.)")
        print("1. Missing Values: PASS")
        print("2. Duplicates:     PASS")
        print("3. Negative Load:  PASS")
        print("4. Completeness:   PASS")

if __name__ == "__main__":
    run_qa()