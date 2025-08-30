import csv
import math
import random
import matplotlib.pyplot as plt

# ----------------------------
# Helper Functions
# ----------------------------

def load_csv(filename):
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            if '' not in row:  # skip rows with missing data
                data.append([float(x) for x in row])
    return data

def train_test_split(data, test_ratio=0.2):
    random.shuffle(data)
    split_index = int(len(data) * (1 - test_ratio))
    return data[:split_index], data[split_index:]

def normalize_features(X):
    mins = [min(col) for col in zip(*X)]
    maxs = [max(col) for col in zip(*X)]
    X_scaled = []
    for row in X:
        X_scaled.append([(row[i] - mins[i]) / (maxs[i] - mins[i]) if maxs[i] != mins[i] else 0 for i in range(len(row))])
    return X_scaled, mins, maxs

def add_bias_column(X):
    return [[1.0] + row for row in X]

# ----------------------------
# Linear Regression from Scratch
# ----------------------------

def predict(X, weights):
    return [sum(w*x for w, x in zip(weights, row)) for row in X]

def compute_cost(X, y, weights):
    predictions = predict(X, weights)
    errors = [(p - t) ** 2 for p, t in zip(predictions, y)]
    return sum(errors) / (2 * len(y))

def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    weights = [0.0] * len(X[0])
    m = len(y)

    for _ in range(epochs):
        predictions = predict(X, weights)
        gradients = [0.0] * len(weights)

        for j in range(len(weights)):
            gradients[j] = sum((predictions[i] - y[i]) * X[i][j] for i in range(m)) / m

        for j in range(len(weights)):
            weights[j] -= learning_rate * gradients[j]

    return weights

# ----------------------------
# Metrics
# ----------------------------

def rmse(y_true, y_pred):
    return math.sqrt(sum((p - t) ** 2 for p, t in zip(y_pred, y_true)) / len(y_true))

def mae(y_true, y_pred):
    return sum(abs(p - t) for p, t in zip(y_pred, y_true)) / len(y_true)

def r2_score(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_tot = sum((t - mean_y) ** 2 for t in y_true)
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    return 1 - (ss_res / ss_tot if ss_tot != 0 else 0)

# ----------------------------
# Run on Multiple Datasets
# ----------------------------

datasets = ["data/boston.csv", "data/california.csv", "data/newyork.csv"]
results = []

for file in datasets:
    data = load_csv(file)
    X = [row[:-1] for row in data]
    y = [row[-1] for row in data]

    X, mins, maxs = normalize_features(X)
    X = add_bias_column(X)

    train_data, test_data = train_test_split(list(zip(X, y)))

    X_train = [row[0] for row in train_data]
    y_train = [row[1] for row in train_data]
    X_test = [row[0] for row in test_data]
    y_test = [row[1] for row in test_data]

    weights = gradient_descent(X_train, y_train, learning_rate=0.01, epochs=1000)
    predictions = predict(X_test, weights)

    results.append({
        "Dataset": file,
        "RMSE": rmse(y_test, predictions),
        "MAE": mae(y_test, predictions),
        "R2": r2_score(y_test, predictions)
    })

# ----------------------------
# Print Results
# ----------------------------

print("\n📊 Linear Regression from Scratch Performance:")
print(f"{'Dataset':<20} {'RMSE':<15} {'MAE':<15} {'R2 Score'}")
for r in results:
    print(f"{r['Dataset']:<20} {r['RMSE']:<15.4f} {r['MAE']:<15.4f} {r['R2']:.4f}")

plt.figure(figsize=(12, 5))

# RMSE Comparison (Log Scale)
plt.subplot(1, 3, 1)
plt.bar([r["Dataset"] for r in results], [r["RMSE"] for r in results], color="skyblue")
plt.title("RMSE Comparison (Log Scale)")
plt.yscale('log')  
plt.xticks(rotation=45, ha="right")

# MAE Comparison (Log Scale)
plt.subplot(1, 3, 2)
plt.bar([r["Dataset"] for r in results], [r["MAE"] for r in results], color="orange")
plt.title("MAE Comparison (Log Scale)")
plt.yscale('log') 
plt.xticks(rotation=45, ha="right")

# R² Comparison
plt.subplot(1, 3, 3)
plt.bar([r["Dataset"] for r in results], [r["R2"] for r in results], color="green")
plt.title("R² Score Comparison")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()
 