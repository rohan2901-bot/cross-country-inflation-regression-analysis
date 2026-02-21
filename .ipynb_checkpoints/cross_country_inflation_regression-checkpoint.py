# ===============================
# IMPORTS
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# ===============================
# LOAD DATA
# ===============================

df = pd.read_csv('clean_data.csv')

print(df.columns)
print(df.head())
print(df.dtypes)
print(df.isnull().sum())
print(df.describe())


# ===============================
# EXPLORATORY DATA ANALYSIS
# ===============================

# Top 20 Countries by Inflation
df_sorted = df.sort_values(by='Inflation Rate (%)', ascending=False)
top20 = df_sorted.head(20)

plt.figure(figsize=(10, 8))
plt.barh(top20['Country'], top20['Inflation Rate (%)'])
plt.xlabel('Inflation Rate (%)')
plt.title('Top 20 Countries by Inflation Rate')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("outputs/top20_inflation.png", dpi=300, bbox_inches='tight')
plt.show()


# Boxplot for outlier detection
plt.figure(figsize=(10, 6))
plt.boxplot(df['Inflation Rate (%)'], vert=False, patch_artist=True)
plt.xlabel('Inflation Rate (%)')
plt.title('Boxplot of Inflation Rates')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("outputs/boxplot_inflation.png", dpi=300, bbox_inches='tight')
plt.show()


# Correlation Matrix
selected_columns = df[['GDP (USD Trillions)', 
                       'Unemployment Rate (%)', 
                       'Inflation Rate (%)']]

correlation_matrix = selected_columns.corr()

print("Correlation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix between GDP, Unemployment Rate, and Inflation Rate')
plt.savefig("outputs/correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()


# Regression plot: Inflation vs Unemployment
plt.figure(figsize=(8, 6))
sns.regplot(x='Unemployment Rate (%)',
            y='Inflation Rate (%)',
            data=df)
plt.title('Inflation vs Unemployment')
plt.show()


# ===============================
# REGRESSION MODELS COMPARISON
# ===============================

# Remove extreme hyperinflation outliers
df_no_outliers = df[df['Inflation Rate (%)'] < 100].copy()

# Log-transform target variable
df_no_outliers['Inflation Rate (%)'] = np.log1p(
    df_no_outliers['Inflation Rate (%)']
)

X = df_no_outliers[['GDP (USD Trillions)', 
                    'Unemployment Rate (%)']]

y = df_no_outliers['Inflation Rate (%)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------------------------------
# Linear Regression
# -------------------------------

lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred_lin = lin_reg.predict(X_test_scaled)

print("Linear Regression Results")
print("R²:", r2_score(y_test, y_pred_lin))
print("MSE:", mean_squared_error(y_test, y_pred_lin))
print("-" * 40)


# -------------------------------
# Ridge Regression
# -------------------------------

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

print("Ridge Regression Results")
print("R²:", r2_score(y_test, y_pred_ridge))
print("MSE:", mean_squared_error(y_test, y_pred_ridge))
print("-" * 40)


# -------------------------------
# Lasso Regression
# -------------------------------

lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)

print("Lasso Regression Results")
print("R²:", r2_score(y_test, y_pred_lasso))
print("MSE:", mean_squared_error(y_test, y_pred_lasso))
print("-" * 40)
