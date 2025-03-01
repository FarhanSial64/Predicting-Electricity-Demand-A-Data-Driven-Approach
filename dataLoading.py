import os
import glob
import pandas as pd
import chardet
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Setup logging
logging.basicConfig(
    filename="data_processing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Define paths
weather_folder = "E:\\DataScience\\Assignment2\\data\\weather_raw_data"
electricity_folder = "E:\\DataScience\\Assignment2\\data\\electricity_raw_data"
json_csv_output_folder = "E:\\DataScience\\Assignment2\\processed_json_data"

# Output files
merged_weather_file = "merged_weather_data.csv"
merged_electricity_file = "merged_electricity_data.csv"

# Ensure output folder exists
os.makedirs(json_csv_output_folder, exist_ok=True)

# Get number of available CPU cores for optimal threading
max_threads = min(10, multiprocessing.cpu_count())  
print(f"🚀 Using {max_threads} threads for fast processing...")
logging.info(f"Using {max_threads} threads for parallel processing.")

### Step 1: Merge all Weather CSV Files ###
def load_csv(file_path):
    """Loads CSV with encoding detection."""
    try:
        with open(file_path, "rb") as f:
            result = chardet.detect(f.read())
        encoding = result["encoding"]
        
        df = pd.read_csv(file_path, encoding=encoding)
        print(f"✅ Successfully loaded: {file_path}")
        logging.info(f"Successfully loaded {file_path}")
        return df
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        logging.error(f"Error loading {file_path}: {e}")
        return None

def process_weather_data():
    """Loads and merges all weather CSV files using multithreading."""
    weather_files = glob.glob(os.path.join(weather_folder, "*.csv"))
    weather_dataframes = []

    print(f"🔄 Merging {len(weather_files)} weather CSV files...")
    with ThreadPoolExecutor(max_threads) as executor:
        future_to_file = {executor.submit(load_csv, file): file for file in weather_files}
        for future in as_completed(future_to_file):
            df = future.result()
            if df is not None:
                weather_dataframes.append(df)

    if weather_dataframes:
        merged_weather_df = pd.concat(weather_dataframes, ignore_index=True, sort=False)
        
        # Print and log dataset info
        print("\n📌 Weather Data Features:", list(merged_weather_df.columns))
        logging.info(f"Weather Data Features: {list(merged_weather_df.columns)}")

        weather_duplicates = merged_weather_df.duplicated().sum()
        print(f"🔍 Duplicate Weather Records Found: {weather_duplicates}")
        logging.info(f"Duplicate Weather Records Found: {weather_duplicates}")

        merged_weather_df.drop_duplicates(inplace=True)

        print(f"✅ Final Weather Data Shape: {merged_weather_df.shape[0]} rows, {merged_weather_df.shape[1]} columns")
        logging.info(f"Final Weather Data Shape: {merged_weather_df.shape}")

        merged_weather_df.to_csv(merged_weather_file, index=False)
        print(f"✅ Merged weather data saved as '{merged_weather_file}'.")
        logging.info(f"Merged weather data saved as '{merged_weather_file}'.")
    else:
        print("⚠️ No valid weather CSV files found.")
        logging.warning("No valid weather CSV files found.")

### Step 2: Convert Each JSON File to CSV ###
def process_json(file_path):
    """Extracts relevant data from a JSON file and saves it as CSV."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        records = json_data.get("response", {}).get("data", [])
        if not records:
            print(f"⚠️ No data found in {file_path}, skipping.")
            logging.warning(f"No data found in {file_path}, skipping.")
            return None
        
        df = pd.DataFrame(records)
        csv_file_path = os.path.join(json_csv_output_folder, os.path.basename(file_path).replace(".json", ".csv"))
        df.to_csv(csv_file_path, index=False)
        print(f"✅ Processed JSON → CSV: {csv_file_path}")
        logging.info(f"Processed and saved {file_path} as {csv_file_path}")
        return df
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        logging.error(f"Error processing {file_path}: {e}")
        return None

def process_electricity_data():
    """Processes all JSON files concurrently using multithreading."""
    electricity_files = glob.glob(os.path.join(electricity_folder, "*.json"))
    processed_files = []

    print(f"🔄 Converting {len(electricity_files)} JSON files to CSV...")
    with ThreadPoolExecutor(max_threads) as executor:
        future_to_file = {executor.submit(process_json, file): file for file in electricity_files}
        for future in as_completed(future_to_file):
            df = future.result()
            if df is not None:
                processed_files.append(df)

    print(f"✅ Processed {len(processed_files)} JSON files into CSVs.")
    logging.info(f"Processed {len(processed_files)} JSON files into CSVs in '{json_csv_output_folder}'.")

### Step 3: Merge All Processed JSON CSVs ###
def merge_electricity_data():
    """Merges all processed JSON CSVs into a single CSV file and renames 'period' to 'date'."""
    processed_csv_files = glob.glob(os.path.join(json_csv_output_folder, "*.csv"))
    processed_dataframes = []

    print(f"🔄 Merging {len(processed_csv_files)} processed JSON CSV files...")
    with ThreadPoolExecutor(max_threads) as executor:
        future_to_file = {executor.submit(pd.read_csv, file): file for file in processed_csv_files}
        for future in as_completed(future_to_file):
            df = future.result()
            if df is not None:
                processed_dataframes.append(df)

    if processed_dataframes:
        merged_electricity_df = pd.concat(processed_dataframes, ignore_index=True, sort=False)

        # Rename 'period' to 'date' if it exists
        if "period" in merged_electricity_df.columns:
            merged_electricity_df.rename(columns={"period": "date"}, inplace=True)
            print("✅ Renamed 'period' column to 'date' in merged electricity dataset.")
            logging.info("Renamed 'period' column to 'date' in merged electricity dataset.")

        # Print and log dataset info
        print("\n📌 Electricity Data Features:", list(merged_electricity_df.columns))
        logging.info(f"Electricity Data Features: {list(merged_electricity_df.columns)}")

        electricity_duplicates = merged_electricity_df.duplicated().sum()
        print(f"🔍 Duplicate Electricity Records Found: {electricity_duplicates}")
        logging.info(f"Duplicate Electricity Records Found: {electricity_duplicates}")

        merged_electricity_df.drop_duplicates(inplace=True)

        print(f"✅ Final Electricity Data Shape: {merged_electricity_df.shape[0]} rows, {merged_electricity_df.shape[1]} columns")
        logging.info(f"Final Electricity Data Shape: {merged_electricity_df.shape}")

        merged_electricity_df.to_csv(merged_electricity_file, index=False)
        print(f"✅ Merged electricity data saved as '{merged_electricity_file}'.")
        logging.info(f"Merged electricity data saved as '{merged_electricity_file}'.")
    else:
        print("⚠️ No valid processed JSON CSV files found.")
        logging.warning("No valid processed JSON CSV files found.")


### **Run All Steps in Order**
if __name__ == "__main__":
    print("🚀 Starting data processing...\n")
    logging.info("🚀 Starting data processing...")

    process_weather_data()  
    process_electricity_data()  
    merge_electricity_data()  

    print("\n✅ Data processing completed successfully!")
    logging.info("✅ Data processing completed successfully!")
