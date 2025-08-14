import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def preprocess_newyork(csv_path: str = "data/newyork.csv"):
   
    df = pd.read_csv(csv_path)

    if 'price' not in df.columns:
        raise ValueError("Expected 'price' column as target in newyork.csv")

    print(f"Original dataset shape: {df.shape}")
    
    # Remove outliers
    df_clean = df[(df['price'] > 0) & (df['price'] < 1000)].copy()
    print(f"After outlier removal: {df_clean.shape}")
    
  
    neighborhood_cols = [col for col in df_clean.columns if 'neighbourhood_group_' in col]
    if neighborhood_cols:
        df_clean['neighbourhood_group'] = df_clean[neighborhood_cols].idxmax(axis=1).str.replace('neighbourhood_group_', '')
        df_clean = df_clean.drop(columns=neighborhood_cols)
    

    room_type_cols = [col for col in df_clean.columns if 'room_type_' in col]
    if room_type_cols:
        df_clean['room_type'] = df_clean[room_type_cols].idxmax(axis=1).str.replace('room_type_', '')
        df_clean = df_clean.drop(columns=room_type_cols)
    

    df_clean['reviews_ratio'] = df_clean['number_of_reviews'] / df_clean['calculated_host_listings_count'].clip(lower=1)
    df_clean['availability_ratio'] = df_clean['availability_365'] / 365
    

    feature_cols = [
        'latitude', 'longitude', 'minimum_nights', 'number_of_reviews',
        'reviews_per_month', 'calculated_host_listings_count', 'availability_365',
        'reviews_ratio', 'availability_ratio'
    ]
    
    feature_cols = [col for col in feature_cols if col in df_clean.columns]
    
    X = df_clean[feature_cols]
    y = df_clean['price']
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    print(f"Final feature set: {len(feature_cols)} features")
    print(f"Features: {feature_cols}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    

    cleaned_path = csv_path.replace('.csv', '_cleaned.csv')
    numeric_df = df_clean[feature_cols + ['price']]  
    numeric_df.to_csv(cleaned_path, index=False)
    print(f"\n💾 Cleaned dataset saved to: {cleaned_path}")
    print(f"   Features saved: {len(feature_cols)} numeric features + price target")
    
    return X_train, X_test, y_train, y_test, feature_cols


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, features = preprocess_newyork()
    print(f"\n✅ Preprocessing complete!")
    print(f"X_train: {np.shape(X_train)}, X_test: {np.shape(X_test)}")
    print(f"y_train: {np.shape(y_train)}, y_test: {np.shape(y_test)}")
    print(f"Number of features: {len(features)}") 