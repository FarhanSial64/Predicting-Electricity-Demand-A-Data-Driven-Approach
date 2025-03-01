# **Predicting Electricity Demand: A Data-Driven Approach**  

## **Project Overview**  
This project focuses on analyzing and predicting electricity demand using real-world data. The dataset includes historical electricity consumption, weather conditions, and time-based features. The goal is to build a robust **data preprocessing and exploratory analysis pipeline**, detect and handle outliers, and develop a **regression model** to predict electricity demand accurately.

## **Project Workflow**  
The project follows a structured workflow:  
1. **Data Cleaning & Preprocessing**  
   - Merging multiple weather CSV files into a single dataset.  
   - Converting electricity demand JSON files into a structured CSV format.  
   - Merging datasets based on timestamps to create a unified dataset.  
   - Extracting useful features such as hour, day, month, and temperature.  

2. **Exploratory Data Analysis (EDA)**  
   - Understanding data distribution and trends.  
   - Identifying missing values and handling them appropriately.  
   - Visualizing relationships between electricity demand and other variables.  

3. **Outlier Detection & Handling**  
   - Detecting outliers using **IQR (Interquartile Range)** and **Z-score method**.  
   - Handling outliers using **capping (5th and 95th percentile)** to retain valuable data.  
   - Visualizing data before and after outlier handling using **box plots and histograms**.  

4. **Regression Modeling for Electricity Demand Prediction**  
   - Selecting relevant features (e.g., temperature, hour, day, month, and day_of_week).  
   - Splitting the dataset into **training and testing sets**.  
   - Training a **Linear Regression model** to predict electricity demand.  
   - Evaluating the model using **MSE, RMSE, and R² score**.  
   - Visualizing **actual vs. predicted values** and performing **residual analysis**.  

## **Technologies Used**  
- **Programming Language:** Python 🐍  
- **Libraries:** pandas, numpy, seaborn, matplotlib, scikit-learn  
- **Visualization Tools:** Matplotlib, Seaborn  
- **Machine Learning Model:** Linear Regression  

## **How to Run the Project**  
1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/FarhanSial64/Predicting-Electricity-Demand-A-Data-Driven-Approach.git
   cd Predicting-Electricity-Demand-A-Data-Driven-Approach
   ```
2. **Install Dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Script:**  
   ```bash
   python main.py
   ```

## **Results & Insights**  
- The model successfully predicts electricity demand based on selected features.  
- **Feature Importance Analysis** shows that time-based variables significantly impact demand patterns.  
- Outlier handling improves model performance by reducing extreme variations.  
- The **residual analysis** suggests areas for further improvement and potential model refinements.  

## **Future Improvements**  
- Experimenting with advanced models like **Random Forest, XGBoost, or LSTMs** for better accuracy.  
- Incorporating additional features like **holiday indicators or economic activity data**.  
- Deploying the model using **Flask or FastAPI** for real-time predictions.  

## **Contributors**  
👤 **Farhan Sial**  
📧 Contact: farhansial64@gmail.com 
🔗 GitHub: [FarhanSial64](https://github.com/FarhanSial64)  

## **License**  
This project is open-source and available under the **MIT License**.  

