import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

def load_data(file_path):
    """Loads the cleaned dataset into a Pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def detect_outliers_iqr(df, column):
    """Detects outliers using the IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def detect_outliers_zscore(df, column, threshold=3):
    """Detects outliers using the Z-score method (default threshold = 3)."""
    z_scores = zscore(df[column])
    outliers = df[np.abs(z_scores) > threshold]
    return outliers

def cap_outliers(df, column):
    """Caps outliers at the 5th and 95th percentiles."""
    lower_cap = df[column].quantile(0.05)
    upper_cap = df[column].quantile(0.95)

    df[column] = np.where(df[column] < lower_cap, lower_cap, df[column])
    df[column] = np.where(df[column] > upper_cap, upper_cap, df[column])

    return df

def plot_before_after_boxplots(df_before, df_after, column):
    """Plots before and after boxplots side by side in the same window."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(y=df_before[column], ax=axes[0], color="lightcoral")
    axes[0].set_title(f"Before Capping - {column}")

    sns.boxplot(y=df_after[column], ax=axes[1], color="lightblue")
    axes[1].set_title(f"After Capping - {column}")

    plt.tight_layout()
    plt.show()

def outlier_analysis(file_path):
    """Performs outlier detection (IQR & Z-score), handling (capping), and visualization."""
    df = load_data(file_path)
    if df is None:
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        print(f"\n🔍 Analyzing Outliers for: {col}")

        # Detect outliers before capping
        outliers_iqr_before = detect_outliers_iqr(df, col)
        outliers_zscore_before = detect_outliers_zscore(df, col)

        print(f"Total Records: {len(df)}")
        print(f"Outliers Before Capping (IQR): {len(outliers_iqr_before)}")
        print(f"Outliers Before Capping (Z-score): {len(outliers_zscore_before)}")

        # Create a copy before capping
        df_before_capping = df.copy()

        # Cap outliers
        df = cap_outliers(df, col)

        # Detect outliers after capping
        outliers_iqr_after = detect_outliers_iqr(df, col)
        outliers_zscore_after = detect_outliers_zscore(df, col)

        print(f"Outliers After Capping (IQR): {len(outliers_iqr_after)}")
        print(f"Outliers After Capping (Z-score): {len(outliers_zscore_after)}")

        # Plot before and after boxplots side by side
        plot_before_after_boxplots(df_before_capping, df, col)

    print("\n✅ Outlier Detection and Handling Completed!")

if __name__ == "__main__":
    file_path = "merged_cleaned_data.csv"
    outlier_analysis(file_path)
