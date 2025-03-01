import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox
from concurrent.futures import ThreadPoolExecutor, as_completed
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Global variable to store DataFrame
df = None

def load_data(file_path):
    """Loads the cleaned dataset into a Pandas DataFrame."""
    global df
    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
        status_label.config(text=f"✅ Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        status_label.config(text=f"❌ Error loading data: {e}")

def statistical_summary():
    """Computes key statistical metrics for numerical features."""
    if df is None:
        status_label.config(text="⚠️ Load data first!")
        return
    numeric_df = df.select_dtypes(include=[np.number])
    summary = numeric_df.describe().T
    summary['skewness'] = numeric_df.skew()
    summary['kurtosis'] = numeric_df.kurt()
    print(summary)
    status_label.config(text="✅ Statistical Summary Computed")

def correlation_analysis():
    """Computes and visualizes the correlation matrix."""
    if df is None:
        status_label.config(text="⚠️ Load data first!")
        return
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    print(correlation_matrix)
    status_label.config(text="✅ Correlation Analysis Done")

def advanced_time_series_analysis():
    """Performs seasonal decomposition and ADF test."""
    if df is None or 'value' not in df.columns:
        status_label.config(text="⚠️ Load data first!")
        return

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    df_sampled = df['value'].resample('h').mean().dropna()
    result = seasonal_decompose(df_sampled, model='additive', period=24)
    adf_test = adfuller(df_sampled.dropna())

    print(f"ADF Statistic: {adf_test[0]}, p-value: {adf_test[1]}")
    status_label.config(text="✅ Time Series Analysis Completed")

def generate_plots():
    """Runs Matplotlib plots in the main thread (Safe for GUI)."""
    if df is None:
        status_label.config(text="⚠️ Load data first!")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['value'], label="Electricity Demand", color='b')
    plt.xlabel("Date")
    plt.ylabel("Electricity Demand")
    plt.title("Electricity Demand Over Time")
    plt.legend()
    plt.show()

    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()
    
    status_label.config(text="✅ Plots Generated")

def start_analysis():
    """Executes data processing tasks in separate threads."""
    if df is None:
        status_label.config(text="⚠️ Load data first!")
        return
    
    status_label.config(text="🚀 Running EDA tasks...")

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_task = {
            executor.submit(statistical_summary): "Statistical Summary",
            executor.submit(correlation_analysis): "Correlation Analysis",
            executor.submit(advanced_time_series_analysis): "Time Series Analysis"
        }
        
        for future in as_completed(future_to_task):
            task_name = future_to_task[future]
            try:
                future.result()
                status_label.after(0, lambda t=task_name: status_label.config(text=f"✅ {t} Done"))
            except Exception as e:
                status_label.after(0, lambda t=task_name, e=e: status_label.config(text=f"❌ {t} Failed: {e}"))

def browse_file():
    """Opens file dialog and loads selected dataset."""
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        load_data(file_path)

# Create Tkinter GUI
root = tk.Tk()
root.title("Electricity Demand EDA")
root.geometry("500x400")

# Buttons
browse_button = tk.Button(root, text="📂 Load Dataset", command=browse_file)
browse_button.pack(pady=5)

run_analysis_button = tk.Button(root, text="🚀 Run EDA", command=start_analysis)
run_analysis_button.pack(pady=5)

plot_button = tk.Button(root, text="📊 Generate Plots", command=generate_plots)
plot_button.pack(pady=5)

# Status Label
status_label = tk.Label(root, text="🔍 Load a dataset to start", fg="blue")
status_label.pack(pady=10)

# Start GUI Event Loop
root.mainloop()
