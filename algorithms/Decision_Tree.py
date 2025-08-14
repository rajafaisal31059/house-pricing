
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


def enhance_newyork_features(df):
    """Create enhanced features for New York dataset to improve Decision Tree performance."""
    df_enhanced = df.copy()
    
    df_enhanced['rooms_rating'] = df_enhanced['rooms'] * df_enhanced['rating']
    df_enhanced['distance_availability'] = df_enhanced['distance_to_center'] * df_enhanced['availability']

    df_enhanced['price_per_room'] = df_enhanced['price'] / df_enhanced['rooms']
    df_enhanced['rating_availability'] = df_enhanced['rating'] / (df_enhanced['availability'] + 0.1)
    
    return df_enhanced



def run_enhanced_decision_tree_on_dataset(dataset_path, target_column, dataset_name, max_depth=None):
    """
    Runs Enhanced Decision Tree Regression on a dataset with feature engineering and hyperparameter tuning.
    """
   
    df = pd.read_csv(dataset_path)
    

    if 'newyork' in dataset_path.lower():
        df = enhance_newyork_features(df)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X = X.fillna(X.mean())
    
    X = X.select_dtypes(include=[np.number])
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    param_grid = {
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Grid search with cross-validation
    dt_base = DecisionTreeRegressor(random_state=42)
    grid_search = GridSearchCV(
        dt_base, param_grid, cv=5, scoring='r2', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    

    best_dt = grid_search.best_estimator_
    

    y_pred = best_dt.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(best_dt, X_train, y_train, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    

    
    return {
        "Dataset": dataset_path,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2
    }



datasets_info = [
    ("data/boston.csv", "MEDV", "Boston Housing"),
    ("data/california.csv", "median_house_value", "California Housing"),
    ("data/newyork.csv", "price", "New York Airbnb")
]



results = []
for dataset_path, target_col, dataset_name in datasets_info:
    res = run_enhanced_decision_tree_on_dataset(dataset_path, target_col, dataset_name)
    results.append(res)


results_df = pd.DataFrame(results)
print("\nDecision Tree Performance Metrics:")
print(results_df)

plt.figure(figsize=(12, 5))

# RMSE Comparison (Log Scale)
plt.subplot(1, 3, 1)
plt.bar(results_df["Dataset"], results_df["RMSE"], color="skyblue")
plt.title("RMSE Comparison (Log Scale)")
plt.yscale('log')  
plt.xticks(rotation=45, ha="right")

# MAE Comparison (Log Scale)
plt.subplot(1, 3, 2)
plt.bar(results_df["Dataset"], results_df["MAE"], color="orange")
plt.title("MAE Comparison (Log Scale)")
plt.yscale('log') 
plt.xticks(rotation=45, ha="right")

# R² Comparison
plt.subplot(1, 3, 3)
plt.bar(results_df["Dataset"], results_df["R²"], color="green")
plt.title("R² Score Comparison")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()


