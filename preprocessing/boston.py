import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_and_preprocess(dataset_name):
 
    if dataset_name == "boston":
        csv_path = "data/boston.csv"
        df = pd.read_csv(csv_path)
    elif dataset_name == "california":
        csv_path = "data/california.csv"
        df = pd.read_csv(csv_path)
    elif dataset_name == "custom":
        csv_path = "data/custom_dataset.csv"
        df = pd.read_csv(csv_path)
    else:
        raise ValueError("Invalid dataset name. Choose from: 'boston', 'california', 'custom'")

    if "MEDV" in df.columns:
        target_col = "MEDV"
    elif "target" in df.columns:
        target_col = "target"
    else:
        raise ValueError("Target column not found in dataset.")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    if dataset_name == "boston":
        processed_df = pd.DataFrame(X_scaled, columns=X.columns)
        processed_df[target_col] = y.values
        processed_df.to_csv(csv_path, index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
