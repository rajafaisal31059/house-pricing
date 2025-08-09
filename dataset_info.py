import pandas as pd
import numpy as np

def analyze_dataset(filename, dataset_name):
    """Analyze and display information about a dataset"""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {dataset_name.upper()} DATASET")
    print(f"File: {filename}")
    print(f"{'='*60}")
    
    try:
        df = pd.read_csv(filename)
        print(f"Dataset Shape: {df.shape}")
        print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        print(f"\nColumn Names:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")
        print(f"\nData Types:")
        print(df.dtypes)
        print(f"\nMissing Values:")
        missing_counts = df.isnull().sum()
        print(missing_counts)
        print(f"Total missing values: {missing_counts.sum()}")
        print(f"\nFirst 3 rows:")
        print(df.head(3))
        print(f"\nBasic Statistics:")
        print(df.describe())
        return True
    except Exception as e:
        print(f"Error reading {filename}: {str(e)}")
        return False

datasets = [
    ('boston.csv', 'Boston Housing'),
    ('newyork.csv', 'New York Housing'),
    ('california.csv', 'California Housing')
]

print("DATASET ANALYSIS REPORT")
print("="*60)

for filename, name in datasets:
    analyze_dataset(filename, name) 