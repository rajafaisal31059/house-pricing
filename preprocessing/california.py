import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def preprocess_california(csv_path: str = "data/california.csv"):
    """Preprocess the California housing dataset and overwrite the CSV.

    Target: 'median_house_value'
    Steps:
    - Impute numeric with median (handles missing in total_bedrooms)
    - One-hot encode 'ocean_proximity'
    - Standardize numeric features
    - Save processed features + target back to csv_path
    - Return train/test splits
    """
    df = pd.read_csv(csv_path)

    if 'median_house_value' not in df.columns:
        raise ValueError("Expected 'median_house_value' as target in california.csv")

    target_col = 'median_house_value'

    categorical_features = ['ocean_proximity'] if 'ocean_proximity' in df.columns else []
    numeric_features = [
        c for c in df.columns if c not in categorical_features + [target_col]
    ]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    X_processed = preprocessor.fit_transform(X)

    num_feature_names = numeric_features
    cat_feature_names = []
    if categorical_features:
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
    feature_names = num_feature_names + cat_feature_names

    if hasattr(X_processed, 'toarray'):
        X_dense = X_processed.toarray()
    else:
        X_dense = X_processed

    processed_df = pd.DataFrame(X_dense, columns=feature_names)
    processed_df[target_col] = y.values

    processed_df.to_csv(csv_path, index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X_dense, y.values, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = preprocess_california()
    print("California preprocessing complete.")
    print(f"X_train: {np.shape(X_train)}, X_test: {np.shape(X_test)}")
    print(f"y_train: {np.shape(y_train)}, y_test: {np.shape(y_test)}") 