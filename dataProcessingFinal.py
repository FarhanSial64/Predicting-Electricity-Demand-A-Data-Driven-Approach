import os
import pandas as pd
import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Setup logging
logging.basicConfig(
    filename="data_cleaning.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# File paths
merged_weather_file = "merged_weather_data.csv"
merged_electricity_file = "merged_electricity_data.csv"
cleaned_weather_file = "cleaned_weather_data.csv"
cleaned_electricity_file = "cleaned_electricity_data.csv"

# Get number of CPU cores for optimal multithreading
max_threads = min(10, multiprocessing.cpu_count())
print(f"🚀 Using {max_threads} threads for efficient processing...")
logging.info(f"Using {max_threads} threads for data cleaning.")

def load_data(file_path):
    """Loads a CSV file into a Pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded {file_path} with {df.shape[0]} records and {df.shape[1]} features.")
        logging.info(f"Loaded {file_path} with {df.shape} shape.")
        return df
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        logging.error(f"Error loading {file_path}: {e}")
        return None

def check_missing_data(df, dataset_name):
    """Identifies missing values and calculates their percentage."""
    missing_percentage = df.isnull().mean() * 100
    missing_summary = missing_percentage[missing_percentage > 0]

    print(f"📉 Missing Data Summary for {dataset_name}:")
    print(missing_summary)
    logging.info(f"Missing Data Summary for {dataset_name}: {missing_summary}")

    return missing_summary

def handle_missing_data(df, dataset_name):
    """Imputes missing data instead of removing it."""
    timestamp_column = "date"

    # Fill missing timestamps using forward-fill and backward-fill
    if timestamp_column in df.columns:
        df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
        df[timestamp_column] = df[timestamp_column].ffill()  # Forward Fill
        df[timestamp_column] = df[timestamp_column].bfill()  # Backward Fill

    # Fixed Imputation for remaining missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])  # Mode for categorical
            else:
                df[col] = df[col].fillna(df[col].median())  # Median for numerical

    print(f"✅ Missing data imputed for {dataset_name}.")
    logging.info(f"Missing data imputed for {dataset_name}.")
    return df

def convert_data_types(df, dataset_name):
    """Converts data types after imputing missing timestamps."""
    timestamp_column = "date"

    if timestamp_column in df.columns:
        df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')

    print(f"✅ Converted {timestamp_column} column to datetime in {dataset_name}.")
    logging.info(f"Converted {timestamp_column} column to datetime in {dataset_name}.")
    return df

def detect_duplicates(df, dataset_name):
    """Detects and removes full duplicate rows."""
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        print(f"✅ Removed {duplicates} duplicate rows from {dataset_name}.")
        logging.info(f"Removed {duplicates} duplicate rows from {dataset_name}.")
    else:
        print(f"✅ No duplicate rows found in {dataset_name}.")
    return df

def detect_outliers(df, dataset_name):
    """Identifies outliers using the IQR method and labels them in a new column."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df["is_outlier"] = "Normal"  # Default label

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
        count = outliers.sum()

        # Print lower and upper bound
        print(f"📊 {col} - Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f} in {dataset_name}")

        if count > 0:
            df.loc[outliers, "is_outlier"] = "Outlier"
            print(f"⚠️ Marked {count} rows as 'Outlier' based on {col} in {dataset_name}.")
            logging.info(f"Marked {count} rows as 'Outlier' based on {col} in {dataset_name}. Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")

    print(f"✅ Outlier detection completed for {dataset_name}.")
    return df

def feature_engineering(df, dataset_name):
    """Creates additional features based on timestamps."""
    timestamp_column = "date"

    if timestamp_column in df.columns:
        df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')

        df['hour'] = df[timestamp_column].dt.hour
        df['day'] = df[timestamp_column].dt.day
        df['month'] = df[timestamp_column].dt.month
        df['day_of_week'] = df[timestamp_column].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

        print("✅ Feature engineering completed.")
        logging.info("Feature engineering completed.")
    return df

def clean_dataset(file_path, output_file, dataset_name):
    """Runs the full data cleaning pipeline on a dataset."""
    df = load_data(file_path)
    if df is not None:
        check_missing_data(df, dataset_name)
        df = handle_missing_data(df, dataset_name)
        df = convert_data_types(df, dataset_name)
        df = detect_duplicates(df, dataset_name)
        df = detect_outliers(df, dataset_name)  # Label outliers instead of removing
        df = feature_engineering(df, dataset_name)
        df.to_csv(output_file, index=False)
        print(f"✅ Cleaned {dataset_name} data saved as {output_file}.")
        logging.info(f"Cleaned {dataset_name} data saved as {output_file}.")

def clean_all_data():
    """Cleans both weather and electricity datasets concurrently."""
    with ThreadPoolExecutor(max_threads) as executor:
        executor.submit(clean_dataset, merged_weather_file, cleaned_weather_file, "Weather")
        executor.submit(clean_dataset, merged_electricity_file, cleaned_electricity_file, "Electricity")

def merge_cleaned_data():
    """Merges cleaned weather and electricity data on the 'date' column."""
    try:
        weather_df = pd.read_csv(cleaned_weather_file)
        electricity_df = pd.read_csv(cleaned_electricity_file)

        # Ensure 'date' column is in datetime format
        weather_df['date'] = pd.to_datetime(weather_df['date'], errors='coerce')
        electricity_df['date'] = pd.to_datetime(electricity_df['date'], errors='coerce')

        # Remove UTC if present
        if weather_df['date'].dtype == "datetime64[ns, UTC]":
            weather_df['date'] = weather_df['date'].dt.tz_localize(None)
        if electricity_df['date'].dtype == "datetime64[ns, UTC]":
            electricity_df['date'] = electricity_df['date'].dt.tz_localize(None)

        # Merge on 'date' column
        merged_df = pd.merge(weather_df, electricity_df, on="date", how="inner")

        # Save the merged dataset
        merged_file = "merged_cleaned_data.csv"
        merged_df.to_csv(merged_file, index=False)

        print(f"✅ Merged cleaned data saved as {merged_file}.")
        logging.info(f"Merged cleaned data saved as {merged_file}.")
    except Exception as e:
        print(f"❌ Error merging cleaned data: {e}")
        logging.error(f"Error merging cleaned data: {e}")



if __name__ == "__main__":
    print("🚀 Starting data cleaning process...")
    logging.info("🚀 Starting data cleaning process...")

    clean_all_data()
    merge_cleaned_data()

    print("✅ Data cleaning completed!")
    logging.info("✅ Data cleaning completed!")









