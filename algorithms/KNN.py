

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



def run_knn_on_dataset(dataset_path, target_column, k=5):
    """
    Runs KNN regression on a dataset and returns evaluation metrics.
    """
    # Load dataset
    df = pd.read_csv(dataset_path)

    # Separate features (X) and target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split into training & testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and train KNN regressor
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predictions
    y_pred = knn.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "Dataset": dataset_path,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2
    }


datasets_info = [
    ("data/boston.csv", "MEDV"),             
    ("data/california.csv", "median_house_value"),
    ("data/newyork.csv", "price")             
]

results = []
for dataset_path, target_col in datasets_info:
    res = run_knn_on_dataset(dataset_path, target_col, k=5)
    results.append(res)

# Create results DataFrame
results_df = pd.DataFrame(results)
print("\n📊 Performance Metrics for Each Dataset:")
print(results_df)


# ===============================
# Plot Comparison Charts
# ===============================
plt.figure(figsize=(12, 5))

# RMSE Comparison
plt.subplot(1, 3, 1)
plt.bar(results_df["Dataset"], results_df["RMSE"], color="skyblue")
plt.title("RMSE Comparison")
plt.xticks(rotation=45, ha="right")

# MAE Comparison
plt.subplot(1, 3, 2)
plt.bar(results_df["Dataset"], results_df["MAE"], color="orange")
plt.title("MAE Comparison")
plt.xticks(rotation=45, ha="right")

# R² Comparison
plt.subplot(1, 3, 3)
plt.bar(results_df["Dataset"], results_df["R²"], color="green")
plt.title("R² Score Comparison")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()
