#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn bits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


OUT = Path("results/insights")
(OUT / "linear_regression").mkdir(parents=True, exist_ok=True)
(OUT / "knn").mkdir(parents=True, exist_ok=True)
(OUT / "decision_tree").mkdir(parents=True, exist_ok=True)
(OUT / "tables").mkdir(parents=True, exist_ok=True)


def evaluate(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2

def scatter_actual_pred(y_true, y_pred, title, path):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    mn, mx = float(np.min(y_true)), float(np.max(y_true))
    plt.plot([mn, mx], [mn, mx], "r--", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def bar_feature_importance(names, values, title, path, top_n=None):
    names = np.array(names)
    values = np.array(values)
    # sort by absolute value descending
    order = np.argsort(np.abs(values))[::-1]
    names = names[order]
    values = values[order]
    if top_n is not None:
        names = names[:top_n]
        values = values[:top_n]
    plt.figure(figsize=(9,6))
    plt.bar(range(len(values)), values, align="center")
    plt.xticks(range(len(values)), names, rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def heatmap_feature_corr(df, target_col, title, path):
    # keep numeric only
    num_df = df.select_dtypes(include=[np.number]).copy()
    # ensure target present
    if target_col not in num_df.columns:
        num_df[target_col] = df[target_col].values
    corr = num_df.corr(numeric_only=True)[[target_col]].sort_values(by=target_col, ascending=False)
    plt.figure(figsize=(6,8))
    plt.imshow(corr.values, aspect="auto")
    plt.colorbar(label="Correlation with target")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.xticks([0], [target_col])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def load_boston():
    df = pd.read_csv("data/boston.csv")
    target = "MEDV"
    X = df.drop(columns=[target])
    y = df[target].values
    features = list(X.columns)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = imputer.fit_transform(X)
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return (X_train, X_test, y_train, y_test), features, df, target

def load_california():
    df = pd.read_csv("data/california.csv")
    target = "median_house_value"
    X = df.drop(columns=[target])
    y = df[target].values

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num_pipe, num_cols),
                             ("cat", cat_pipe, cat_cols)])
    X_trans = pre.fit_transform(X)

    
    names_num = num_cols
    names_cat = []
    if len(cat_cols) > 0:
        ohe = pre.named_transformers_["cat"].named_steps["onehot"]
        names_cat = ohe.get_feature_names_out(cat_cols).tolist()
    features = names_num + names_cat

    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.2, random_state=42)
    return (X_train, X_test, y_train, y_test), features, df, target

def load_newyork():
    df = pd.read_csv("data/newyork.csv")
    # simple clean
    df = df[df["price"] > 0].copy()
    target = "price"
    X = df.drop(columns=[target])
    y = df[target].values
    features = list(X.columns)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = imputer.fit_transform(X)
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return (X_train, X_test, y_train, y_test), features, df, target

DATASETS = {
    "boston": load_boston,
    "california": load_california,
    "newyork": load_newyork
}


def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

def train_linear_gd(X, y, lr=0.01, epochs=2000):
    Xb = add_bias(X)
    n, d = Xb.shape
    w = np.zeros(d)
    for _ in range(epochs):
        y_hat = Xb @ w
        grad = (2.0 / n) * Xb.T @ (y_hat - y)
        w -= lr * grad
    return w  
def predict_linear(X, w):
    return add_bias(X) @ w

def linear_regression_plots():
    rows = []
    for dsname, loader in DATASETS.items():
        (X_train, X_test, y_train, y_test), features, df_raw, target = loader()

        # fit
        w = train_linear_gd(X_train, y_train, lr=0.01, epochs=2000)
        y_pred = predict_linear(X_test, w)
        rmse, mae, r2 = evaluate(y_test, y_pred)
        rows.append({"Algorithm": "Linear Regression",
                     "Dataset": dsname, "RMSE": rmse, "MAE": mae, "R2": r2})

        # scatter
        scatter_actual_pred(
            y_test, y_pred,
            f"Linear Regression - {dsname.capitalize()} (Actual vs Predicted)",
            OUT / "linear_regression" / f"scatter_{dsname}.png"
        )


        coef = w[1:]
     
        top_n = 15 if len(features) > 20 else None
        bar_feature_importance(
            features, coef,
            f"Linear Regression Feature Importance - {dsname.capitalize()}",
            OUT / "linear_regression" / f"feature_importance_{dsname}.png",
            top_n=top_n
        )

    pd.DataFrame(rows).to_csv(OUT / "tables" / "linear_regression_metrics.csv", index=False)

def knn_plots():
    rows = []
    for dsname, loader in DATASETS.items():
        (X_train, X_test, y_train, y_test), features, df_raw, target = loader()

       
        ks = list(range(1, 26))  
        rmses = []
        for k in ks:
            model = KNeighborsRegressor(n_neighbors=k)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse, _, _ = evaluate(y_test, preds)
            rmses.append(rmse)

        best_idx = int(np.argmin(rmses))
        best_k = ks[best_idx]


        model = KNeighborsRegressor(n_neighbors=best_k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse, mae, r2 = evaluate(y_test, y_pred)
        rows.append({"Algorithm": "KNN",
                     "Dataset": dsname, "BestK": best_k, "RMSE": rmse, "MAE": mae, "R2": r2})

        # K optimization curve
        plt.figure(figsize=(7,5))
        plt.plot(ks, rmses, marker="o")
        plt.xlabel("K value")
        plt.ylabel("RMSE")
        plt.title(f"K optimization on {dsname.capitalize()}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / "knn" / f"k_optimization_{dsname}.png")
        plt.close()

        # feature correlation heatmap
        heatmap_feature_corr(
            df_raw, target,
            f"Feature correlation with target - {dsname.capitalize()}",
            OUT / "knn" / f"feature_corr_{dsname}.png"
        )

    pd.DataFrame(rows).to_csv(OUT / "tables" / "knn_metrics.csv", index=False)


def decision_tree_plots():
    rows = []
    for dsname, loader in DATASETS.items():
        (X_train, X_test, y_train, y_test), features, df_raw, target = loader()

        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse, mae, r2 = evaluate(y_test, y_pred)
        rows.append({"Algorithm": "Decision Tree",
                     "Dataset": dsname, "RMSE": rmse, "MAE": mae, "R2": r2})

        # scatter
        scatter_actual_pred(
            y_test, y_pred,
            f"Decision Tree - {dsname.capitalize()} (Actual vs Predicted)",
            OUT / "decision_tree" / f"scatter_{dsname}.png"
        )

        # feature importance
        if features is not None:
            importances = model.feature_importances_
            top_n = 15 if len(features) > 20 else None
            bar_feature_importance(
                features, importances,
                f"Decision Tree Feature Importance - {dsname.capitalize()}",
                OUT / "decision_tree" / f"feature_importance_{dsname}.png",
                top_n=top_n
            )

    pd.DataFrame(rows).to_csv(OUT / "tables" / "decision_tree_metrics.csv", index=False)


def main():
    print("Generating Linear Regression figures and table...")
    linear_regression_plots()
    print("Generating KNN figures and table...")
    knn_plots()
    print("Generating Decision Tree figures and table...")
    decision_tree_plots()
    print(f"Done. See {OUT.resolve()}")

if __name__ == "__main__":
    main()
