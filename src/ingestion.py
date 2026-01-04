import requests
import pandas as pd
import xml.etree.ElementTree as ET
import re
import os
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

# CONFIGURATION
# Read from environment with sensible fallbacks
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
OUTPUT_PATH = "data/submission_data.csv"

# NETHERLANDS (NL) - Robust Single Code
COUNTRY_CONFIG = {
    "name": "Netherlands",
    "code": "10YNL----------L" 
}

START_TIME = "202301010000"
END_TIME   = "202401010000"

# Ensure output directory exists
os.makedirs("data", exist_ok=True)

def fetch_entsoe(doc_type):
    params = {
        "securityToken": API_KEY,
        "documentType": doc_type,
        "periodStart": START_TIME,
        "periodEnd": END_TIME
    }
    
    # NL Specific Logic
    if doc_type == "A44": # Prices
        params["in_Domain"] = COUNTRY_CONFIG["code"]
        params["out_Domain"] = COUNTRY_CONFIG["code"]
        params["contract_MarketAgreement.Type"] = "A01"
    elif doc_type == "A65": # Load
        params["outBiddingZone_Domain"] = COUNTRY_CONFIG["code"]
        params["processType"] = "A01"
    elif doc_type == "A69": # Gen
        params["in_Domain"] = COUNTRY_CONFIG["code"]
        params["processType"] = "A01"

    try:
        response = requests.get(BASE_URL, params=params, timeout=120)
        if response.status_code == 200 and "No matching data" not in response.text:
            print(f"   ✅ Success: {doc_type} ({len(response.text)/1024:.1f} KB)")
            return response.text
        print(f"   ❌ Error: {doc_type} | Code: {response.status_code}")
        return None
    except Exception as e:
        print(f"   ❌ Network Error: {e}")
        return None

def parse_xml_robust(xml_str, val_tag):
    if not xml_str: return pd.DataFrame()
    xml_str = re.sub(r'xmlns="[^"]+"', '', xml_str) # Strip namespaces
    try:
        root = ET.fromstring(xml_str)
    except:
        return pd.DataFrame()
        
    data = []
    for ts in root.findall(".//TimeSeries"):
        psr_type = ts.find(".//MktPSRType/psrType")
        psr_code = psr_type.text if psr_type is not None else "General"
        
        start_node = ts.find(".//period.timeInterval/start")
        if start_node is None: start_node = ts.find(".//Period/timeInterval/start")
        if start_node is None: continue
        start_dt = pd.to_datetime(start_node.text)
        
        res_node = ts.find(".//Period/resolution")
        res_min = 15 if res_node is not None and res_node.text == 'PT15M' else 60
        
        for point in ts.findall(".//Period/Point"):
            pos = int(point.find("position").text)
            val = float(point.find(val_tag).text)
            timestamp = start_dt + pd.Timedelta(minutes=(pos-1)*res_min)
            data.append({"Timestamp_UTC": timestamp, "Value": val, "Type": psr_code})
            
    return pd.DataFrame(data)

print(f"🚀 Starting Ingestion: {COUNTRY_CONFIG['name']}")

# Fetch
xml_price = fetch_entsoe("A44")
df_price = parse_xml_robust(xml_price, "price.amount")

xml_load = fetch_entsoe("A65")
df_load = parse_xml_robust(xml_load, "quantity")

xml_gen = fetch_entsoe("A69")
df_gen_raw = parse_xml_robust(xml_gen, "quantity")

# Process
if not df_price.empty and not df_load.empty:
    print("⚙️ Processing...")
    df_price = df_price.set_index("Timestamp_UTC")[["Value"]].rename(columns={"Value": "Price_EUR"}).resample('h').mean()
    df_load = df_load.set_index("Timestamp_UTC")[["Value"]].rename(columns={"Value": "Load_MW"}).resample('h').mean()
    
    if not df_gen_raw.empty:
        type_map = {"B16": "Solar", "B19": "Wind_Onshore", "B18": "Wind_Offshore"}
        df_gen_raw["Type"] = df_gen_raw["Type"].map(type_map)
        df_gen = df_gen_raw.pivot_table(index="Timestamp_UTC", columns="Type", values="Value", aggfunc='mean').resample('h').mean()
        df_final = df_price.join(df_load, how="inner").join(df_gen, how="left").fillna(0)
    else:
        df_final = df_price.join(df_load, how="inner")

    df_final.to_csv(OUTPUT_PATH)
    print(f"✨ FINAL SUCCESS! Saved to '{OUTPUT_PATH}'")
    print(df_final.head())
else:
    print("❌ Failed to merge data.")