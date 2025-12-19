import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import logging
import os
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HeartDiseasePreprocessor:
    def __init__(self, raw_data_path='heart_disease.csv', output_path='preprocessing/heart_preprocessing'):
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)

    def load_data(self):
        #Load raw data from CSV file
        logger.info(f"Loading data from {self.raw_data_path}")
        self.df = pd.read_csv(self.raw_data_path)
        logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
        return self

    def check_data_quality(self):
        logger.info("Checking data quality...")

        # Check Missing values
        missing = self.df.isnull().sum().sum()
        logger.info(f"Total missing values: {missing}")

        # Check Duplicates
        duplicates = self.df.duplicated().sum()
        logger.info(f"Total duplicate rows: {duplicates}")

        return self

    def handle_missing_values(self):
        logger.info("Handling missing values...")

        # Drop Alcohol Consumption column (if exists)
        if "Alcohol Consumption" in self.df.columns:
            self.df = self.df.drop(columns=["Alcohol Consumption"])
            logger.info("Dropped 'Alcohol Consumption' column")

        # Drop remaining missing values
        initial_rows = self.df.shape[0]
        self.df = self.df.dropna().reset_index(drop=True)
        removed_rows = initial_rows - self.df.shape[0]

        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} rows with missing values")
        else:
            logger.info("No missing values found")

        return self

    def remove_duplicates(self):
        initial_rows = self.df.shape[0]
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        removed = initial_rows - self.df.shape[0]

        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        else:
            logger.info("No duplicate rows found")

        return self

    def detect_outliers(self):
        # Detect outliers using IQR method
        logger.info("Detecting outliers...")

        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        outlier_report = {}

        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)]
            outlier_report[col] = len(outliers)

        logger.info("Outlier detection completed:")
        for col, count in outlier_report.items():
            if count > 0:
                logger.info(f"  {col}: {count} outliers")

        return self

    def feature_engineering(self):
        logger.info("Starting feature engineering...")

        # Create age groups
        if "Age" in self.df.columns:
            bins = [18, 34, 49, 64, 80]
            labels = ["young_adult", "adult", "middle_aged", "senior"]

            self.df["age_group"] = pd.cut(
                self.df["Age"],
                bins=bins,
                labels=labels,
                include_lowest=True
            )
            logger.info("Created 'age_group' feature")
        return self

    def split_features_target(self):
        logger.info("Splitting features and target...")

        target_col = "Heart Disease Status"
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")

        self.X = self.df.drop(target_col, axis=1)
        self.y = self.df[target_col]

        logger.info(f"Features shape: {self.X.shape}")
        logger.info(f"Target shape: {self.y.shape}")
        logger.info(f"Target distribution:\n{self.y.value_counts()}")

        return self

    def train_test_split_data(self, test_size=0.2, random_state=42):
        logger.info(f"Splitting data (test_size={test_size}, stratified)...")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y
        )

        logger.info(f"Training set: {self.X_train.shape}")
        logger.info(f"Test set: {self.X_test.shape}")

        return self

    def encode_age_group(self):
        logger.info("Encoding age_group...")

        if "age_group" in self.X_train.columns:
            # One-hot encoding
            self.X_train = pd.get_dummies(
                self.X_train,
                columns=["age_group"],
                drop_first=True
            )
            self.X_test = pd.get_dummies(
                self.X_test,
                columns=["age_group"],
                drop_first=True
            )

            # Align columns (handle missing categories in test set)
            self.X_train, self.X_test = self.X_train.align(
                self.X_test,
                join="left",
                axis=1,
                fill_value=0
            )

            logger.info("✓ Age group encoded with one-hot encoding")

        return self

    def encode_boolean_columns(self):
        logger.info("Detecting and encoding boolean columns...")

        bool_like_cols = []

        # Detect boolean-like columns
        for col in self.X_train.columns:
            unique_vals = set(
                self.X_train[col].dropna().astype(str).str.lower().str.strip()
            )
            if unique_vals.issubset({"true", "false"}) or unique_vals.issubset({"yes", "no"}):
                bool_like_cols.append(col)

        # Encode boolean columns
        bool_map = {"true": 1, "false": 0, "yes": 1, "no": 0}

        for col in bool_like_cols:
            self.X_train[col] = (
                self.X_train[col]
                .astype(str)
                .str.lower()
                .str.strip()
                .map(bool_map)
                .astype(int)
            )
            self.X_test[col] = (
                self.X_test[col]
                .astype(str)
                .str.lower()
                .str.strip()
                .map(bool_map)
                .astype(int)
            )

        if bool_like_cols:
            logger.info(f"✓ Encoded boolean columns: {bool_like_cols}")
        else:
            logger.info("✓ No boolean columns detected")

        return self

    def encode_gender(self):
        logger.info("Encoding Gender column...")

        if "Gender" in self.X_train.columns:
            gender_map = {"male": 1, "female": 0}

            self.X_train["Gender"] = (
                self.X_train["Gender"]
                .astype(str)
                .str.lower()
                .str.strip()
                .map(gender_map)
                .astype(int)
            )
            self.X_test["Gender"] = (
                self.X_test["Gender"]
                .astype(str)
                .str.lower()
                .str.strip()
                .map(gender_map)
                .astype(int)
            )

            logger.info("Gender encoded (Male=1, Female=0)")
        else:
            logger.info("Gender column not found")

        return self
        
    def encode_target(self):
        logger.info("Encoding target variable (y)...")
    
        target_map = {"yes": 1, "no": 0}
    
        # Encode y_train
        self.y_train = (
            self.y_train
            .astype(str)
            .str.lower()
            .str.strip()
            .map(target_map)
        )
    
        # Encode y_test
        self.y_test = (
            self.y_test
            .astype(str)
            .str.lower()
            .str.strip()
            .map(target_map)
        )
    
        # Validasi NaN
        nan_train = self.y_train.isna().sum()
        nan_test = self.y_test.isna().sum()
    
        if nan_train > 0 or nan_test > 0:
            logger.warning(
                f"NaN detected after target encoding "
                f"(train={nan_train}, test={nan_test})"
            )
    
        logger.info("Target encoded (Yes=1, No=0)")
        return self


    def encode_ordinal_columns(self):
        logger.info("Detecting and encoding ordinal columns...")

        ordinal_cols = []

        # Detect ordinal columns
        for col in self.X_train.columns:
            unique_vals = set(
                self.X_train[col].dropna().astype(str).str.lower().str.strip()
            )
            if unique_vals.issubset({"low", "medium", "high"}):
                ordinal_cols.append(col)

        # Encode ordinal columns
        ordinal_map = {"low": 0, "medium": 1, "high": 2}

        for col in ordinal_cols:
            self.X_train[col] = (
                self.X_train[col]
                .astype(str)
                .str.lower()
                .str.strip()
                .map(ordinal_map)
                .astype(int)
            )
            self.X_test[col] = (
                self.X_test[col]
                .astype(str)
                .str.lower()
                .str.strip()
                .map(ordinal_map)
                .astype(int)
            )

        if ordinal_cols:
            logger.info(f"Encoded ordinal columns: {ordinal_cols}")
        else:
            logger.info("No ordinal columns detected")

        return self

    def scale_features(self):
        logger.info("Scaling numeric features...")

        # Identify numeric columns
        numeric_cols = self.X_train.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        logger.info(f"Numeric columns to scale: {len(numeric_cols)}")

        # Create copies
        self.X_train_scaled = self.X_train.copy()
        self.X_test_scaled = self.X_test.copy()

        # Scale only numeric columns
        if numeric_cols:
            self.X_train_scaled[numeric_cols] = self.scaler.fit_transform(
                self.X_train[numeric_cols]
            )
            self.X_test_scaled[numeric_cols] = self.scaler.transform(
                self.X_test[numeric_cols]
            )

            logger.info("Numeric features scaled with StandardScaler")
        else:
            logger.info("No numeric columns to scale")

        return self

    def apply_smote(self):
        logger.info("Applying SMOTE for class balancing...")

        # Check class distribution before SMOTE
        before_counts = self.y_train.value_counts()
        logger.info(f"Before SMOTE:\n{before_counts}")

        # Apply SMOTE
        self.X_train_final, self.y_train_final = self.smote.fit_resample(
            self.X_train_scaled, 
            self.y_train
        )

        # Check class distribution after SMOTE
        after_counts = self.y_train_final.value_counts()
        logger.info(f"After SMOTE:\n{after_counts}")

        # Test set remains unchanged
        self.X_test_final = self.X_test_scaled
        self.y_test_final = self.y_test

        logger.info("SMOTE applied successfully")

        return self

    def save_preprocessed_data(self):
        logger.info(f"Saving preprocessed data to '{self.output_path}/'...")
    
        os.makedirs(self.output_path, exist_ok=True)
    
        files = {
            "X_train_preprocessing.csv": self.X_train_final,
            "y_train_preprocessing.csv": self.y_train_final,
            "X_test_preprocessing.csv": self.X_test_final,
            "y_test.csv": self.y_test_final,
        }
    
        for filename, data in files.items():
            file_path = os.path.join(self.output_path, filename)
    
            if os.path.exists(file_path):
                logger.info(f"Updating existing file: {filename}")
            else:
                logger.info(f"Creating new file: {filename}")
    
            data.to_csv(file_path, index=False)
    
        logger.info("All files saved/updated successfully!")
        logger.info(f"  - X_train_preprocessing.csv: {self.X_train_final.shape}")
        logger.info(f"  - y_train_preprocessing.csv: {self.y_train_final.shape}")
        logger.info(f"  - X_test_preprocessing.csv: {self.X_test_final.shape}")
        logger.info(f"  - y_test.csv: {self.y_test_final.shape}")
    
        return self



    def run_pipeline(self):
        logger.info("STARTING AUTOMATED PREPROCESSING PIPELINE")

        try:
            self.load_data()
            self.check_data_quality()
            self.handle_missing_values()
            self.remove_duplicates()
            self.detect_outliers()
            self.feature_engineering()
            self.split_features_target()
            self.train_test_split_data()
            self.encode_age_group()
            self.encode_boolean_columns()
            self.encode_gender()
            self.encode_ordinal_columns()
            self.encode_target()
            self.scale_features()
            self.apply_smote()
            self.save_preprocessed_data()

            logger.info("="*60)
            logger.info("PREPROCESSING COMPLETED SUCCESSFULLY!")

            # Print summary
            self._print_summary()

            return True

        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            raise

    def _print_summary(self):
        print("PREPROCESSING SUMMARY")
        print(f"Raw data shape:          {self.df.shape}")
        print(f"Train set (after SMOTE): {self.X_train_final.shape}")
        print(f"Test set:                {self.X_test_final.shape}")
        print(f"Number of features:      {self.X_train_final.shape[1]}")
        print(f"Target classes:          {self.y_train_final.nunique()}")
        print("\nTarget distribution (train after SMOTE):")
        print(self.y_train_final.value_counts())
        print("\nTarget distribution (test):")
        print(self.y_test_final.value_counts())
        print("Data is ready for training!")


def main():
    # Configuration
    RAW_DATA_PATH = 'heart_disease.csv'
    OUTPUT_PATH = "preprocessing/heart_preprocessing"
    # Initialize and run preprocessor
    preprocessor = HeartDiseasePreprocessor(RAW_DATA_PATH, OUTPUT_PATH)
    preprocessor.run_pipeline()

if __name__ == "__main__":
    main()
