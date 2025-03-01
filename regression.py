import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("merged_cleaned_data.csv")

# Drop unnecessary columns
df = df[['temperature_2m', 'hour_x', 'day_x', 'month_x', 'day_of_week_x', 'is_weekend_x', 'value']]
df.dropna(inplace=True)

# Outlier Detection & Capping
def cap_outliers(data, column):
    Q1 = data[column].quantile(0.05)
    Q3 = data[column].quantile(0.95)
    data[column] = np.clip(data[column], Q1, Q3)

for col in ['temperature_2m', 'value']:
    cap_outliers(df, col)

# Feature Scaling
scaler = StandardScaler()
df[['temperature_2m', 'hour_x', 'day_x', 'month_x', 'day_of_week_x']] = scaler.fit_transform(
    df[['temperature_2m', 'hour_x', 'day_x', 'month_x', 'day_of_week_x']]
)

# Train-Test Split
X = df.drop(columns=['value'])
y = df['value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Evaluation
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Performance:")
    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}\n")

evaluate_model(y_test, lr_preds, "Linear Regression")
evaluate_model(y_test, rf_preds, "Random Forest")

# Visualization
plt.figure(figsize=(10, 5))
plt.scatter(y_test, lr_preds, label='Linear Regression', alpha=0.5)
plt.scatter(y_test, rf_preds, label='Random Forest', alpha=0.5)
plt.plot(y_test, y_test, color='red', linestyle='--', label='Ideal Fit')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Electricity Demand")
plt.legend()
plt.show()

# Residual Analysis
residuals = y_test - rf_preds
sns.histplot(residuals, kde=True)
plt.title("Residual Analysis")
plt.show()
