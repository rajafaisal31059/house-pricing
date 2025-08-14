import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

boston_df = pd.read_csv('data/boston.csv')

print("=== BOSTON DATASET STRUCTURE (from boston.csv) ===")
print(f"Dataset shape: {boston_df.shape}")
print(f"\nColumn names:")
for i, col in enumerate(boston_df.columns):
    print(f"{i+1:2d}. {col}")
print(f"\nFirst 5 rows:")
print(boston_df.head())
print(f"\nData types:")
print(boston_df.dtypes)
print(f"\nBasic statistics:")
print(boston_df.describe())
print(f"\nMissing values:")
print(boston_df.isnull().sum())
print("=" * 50)

X = boston_df.iloc[:, :-1]
y = boston_df.iloc[:, -1]

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target variable name: {y.name}")

print(f"\nMissing values in features before imputation:")
print(X.isnull().sum())

imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print(f"\nMissing values in features after imputation:")
print(X_imputed.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n=== MODEL RESULTS ===")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("KNN: Actual vs Predicted Prices")
plt.grid(True, alpha=0.3)
plt.show()
