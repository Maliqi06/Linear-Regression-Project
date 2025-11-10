EV Battery Charging Efficiency – Linear Regression Analysis

This project analyzes synthetic electric vehicle (EV) battery charging data using Linear Regression. It includes data cleaning, visualization, model training, evaluation, and prediction to understand how different parameters influence charging efficiency.

📌 Features of the Project

Upload and load synthetic dataset in Google Colab

Clean and preprocess battery charging data

Encode categorical variables

Visualize relationships between key features

Train a Linear Regression model

Compare actual vs predicted efficiency

Evaluate performance using:

Mean Squared Error (MSE)

R-Squared (R²)

📂 Files

ev_battery_charging_data.csv – Synthetic dataset used in this project

battery_efficiency_model.ipynb / .py – Main analysis code

README (this file)

🚀 How It Works

Upload Dataset
Uses google.colab.files.upload() to import the CSV file.

Data Cleaning

Removes additional spaces

Converts categorical text into numeric labels

Exploratory Data Analysis (EDA)

Pairplot visualizations with Seaborn

Model Training

Splits data into training and test sets

Fits a Linear Regression model

Model Evaluation

Displays coefficients

Plots actual vs predicted

Calculates MSE & R²

📊 Example Outputs

Unique category values

Regression coefficients

Actual vs predicted efficiency plot

Model performance metrics

✅ Requirements

Install dependencies:

pip install pandas matplotlib seaborn scikit-learn

🧠 Technologies Used

Python

Pandas

Matplotlib & Seaborn

Scikit-learn

Google Colab
