import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
import os
import sys

warnings.filterwarnings('ignore')


def load_raw_data(filepath):
    print("LOADING DATA")
    print(f"Loading data from: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)
    print("Data loaded successfully")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    return df


def check_data_quality(df):
    print("DATA QUALITY CHECK")

    missing = df.isnull().sum()
    missing_total = missing.sum()

    print(f"\nMissing Values: {missing_total}")
    if missing_total > 0:
        print("\nMissing values per column:")
        for col, count in missing[missing > 0].items():
            pct = (count / len(df)) * 100
            print(f"  {col}: {count} ({pct:.2f}%)")
    else:
        print("No missing values found")

    duplicate_count = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicate_count}")

    print("\nData types:")
    print(df.dtypes.value_counts())

    target_col = df.columns[-1]
    print(f"\nTarget column: {target_col}")
    print("Target distribution:")
    print(df[target_col].value_counts())


def handle_missing_values(df):
    print("HANDLING MISSING VALUES")

    if df.isnull().sum().sum() == 0:
        print("No missing values to handle")
        return df

    df_clean = df.copy()

    numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()

    if numeric_cols:
        imputer_numeric = SimpleImputer(strategy='median')
        df_clean[numeric_cols] = imputer_numeric.fit_transform(df_clean[numeric_cols])

    if categorical_cols:
        imputer_categorical = SimpleImputer(strategy='most_frequent')
        df_clean[categorical_cols] = imputer_categorical.fit_transform(df_clean[categorical_cols])

    return df_clean


def remove_duplicates(df):
    print("REMOVING DUPLICATES")
    return df.drop_duplicates().reset_index(drop=True)


def handle_outliers(df, target_column=None):
    print("HANDLING OUTLIERS")

    if target_column is None:
        target_column = df.columns[-1]

    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [c for c in numeric_cols if c != target_column]

    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 2.5 * IQR
        upper = Q3 + 2.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]

    return df_clean.reset_index(drop=True)


def encode_categorical(df, target_column=None):
    print("ENCODING CATEGORICAL VARIABLES")

    if target_column is None:
        target_column = df.columns[-1]

    df_encoded = df.copy()

    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    feature_cats = [c for c in categorical_cols if c != target_column]

    if feature_cats:
        df_encoded = pd.get_dummies(df_encoded, columns=feature_cats, drop_first=True)

    if target_column in categorical_cols:
        le = LabelEncoder()
        df_encoded[target_column] = le.fit_transform(df_encoded[target_column])

    return df_encoded


def scale_features(df, target_column=None):
    print("SCALING FEATURES")

    if target_column is None:
        target_column = df.columns[-1]

    df_scaled = df.copy()
    features = [c for c in df_scaled.columns if c != target_column]

    scaler = StandardScaler()
    df_scaled[features] = scaler.fit_transform(df_scaled[features])

    return df_scaled


def save_preprocessed_data(df, output_path):
    print("SAVING PREPROCESSED DATA")
    df.to_csv(output_path, index=False)
    print(f"Data saved to: {output_path}")
    print(f"Final shape: {df.shape}")


def preprocess_heart_disease(input_file, output_file):
    print("\nHEART DISEASE DATA PREPROCESSING PIPELINE")

    df = load_raw_data(input_file)
    check_data_quality(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = handle_outliers(df)
    df = encode_categorical(df)
    df = scale_features(df)
    save_preprocessed_data(df, output_file)

    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    return df


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    input_file = os.path.join(project_root, 'heart_disease.csv')
    output_file = os.path.join(script_dir, 'heart_disease_preprocessing.csv')

    if not os.path.exists(input_file):
        print("ERROR: Input file not found")
        print("Expected file location:")
        print(input_file)
        return 1

    try:
        preprocess_heart_disease(input_file, output_file)
        return 0
    except Exception as e:
        print(f"ERROR during preprocessing: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
