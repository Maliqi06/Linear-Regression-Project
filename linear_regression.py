# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from google.colab import files

# -------------------------------------------------------
# 1. Upload and Load Dataset
# -------------------------------------------------------
uploaded = files.upload()  # Allows selecting file in Colab

# Load CSV file into pandas DataFrame
df = pd.read_csv('ev_battery_charging_data.csv')

# -------------------------------------------------------
# 2. Data Cleaning
# -------------------------------------------------------
# Convert string columns to clean text (remove trailing spaces)
df['Battery Type'] = df['Battery Type'].astype(str).str.strip()
df['Charging Mode'] = df['Charging Mode'].astype(str).str.strip()

# Display unique category values
print("Unique Battery Types:", df['Battery Type'].unique())
print("Unique Charging Modes:", df['Charging Mode'].unique())
print("Unique EV Models:", df['EV Model'].unique())

# Convert categorical text values into numeric (Label Encoding)
df['Battery Type'] = df['Battery Type'].map({'Li-ion': 0, 'LiFePO4': 1})
df['Charging Mode'] = df['Charging Mode'].map({'Fast': 0, 'Slow': 1, 'Normal': 2})
df['EV Model'] = df['EV Model'].map({'Model B': 0, 'Model A': 1, 'Model C': 2})

# -------------------------------------------------------
# 3. Define Features and Target
# -------------------------------------------------------
X = df.drop("Efficiency (%)", axis=1)   # Independent variables
y = df["Efficiency (%)"]                # Target variable

# Check for missing values
print(df.isnull().sum())

# Display dataset summary
print(df.info())
print(df.describe())

# -------------------------------------------------------
# 4. Explore Relationships (Correlation Visualization)
# -------------------------------------------------------
sns.pairplot(df, vars=[
    "SOC (%)", "Voltage (V)", "Current (A)",
    "Battery Temp (°C)", "Ambient Temp (°C)",
    "Charging Duration (min)", "Degradation Rate (%)",
    "Efficiency (%)", "Charging Cycles", "EV Model"
])
plt.show()

# -------------------------------------------------------
# 5. Split Data into Training and Testing Sets
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------------
# 6. Train Linear Regression Model
# -------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# Predict target values for test data
y_pred = model.predict(X_test)

# Display regression coefficients
print("Slope (Coefficients):", model.coef_)
print("Intercept:", model.intercept_)

# -------------------------------------------------------
# 7. Visualize Actual vs Predicted Values
# -------------------------------------------------------
plt.scatter(y_test, y_pred)
max_val = max(y_test.max(), y_pred.max())
min_val = min(y_test.min(), y_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')
plt.title("Actual vs Predicted Efficiency (%)")
plt.xlabel("Actual Efficiency (%)")
plt.ylabel("Predicted Efficiency (%)")
plt.grid(True)
plt.show()

# -------------------------------------------------------
# 8. Evaluate Model Performance
# -------------------------------------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-Squared (R²):", r2)
