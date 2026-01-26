import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import Core_Operations
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataPreProcessing:

    def __init__(self):
        # Memory for Encoders
        self.label_encoders = {}
        # Memory for Scaler
        self.scaler = StandardScaler()
        # Memory for Mean Imputation
        self.mean_total_charges = 0
        self.training_features = None

    def preprocess_data(self, df: pd.DataFrame, target_col: str = "Churn", is_training: bool = True) -> pd.DataFrame:
        """
        is_training=True  -> Fits encoders & scaler on data.
        is_training=False -> Applies saved encoders & scaler to new data.
        """
        df = df.copy()
        df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace
        if is_training:
            df = df.drop_duplicates(keep='last')
        
        service_col=["StreamingTV", "StreamingMovies", "InternetService", "MultipleLines"]
        existing_service_cols = [c for c in service_col if c in df.columns]
        if existing_service_cols:
            df["n_services"] = (df[existing_service_cols] == "Yes").sum(axis=1)
            df = df.drop(columns=existing_service_cols)
        
        for col in ["customerID", "CustomerID", "customer_id"]:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # target to 0/1 if it's Yes/No
        if target_col in df.columns and df[target_col].dtype == "object":
            df[target_col] = df[target_col].str.strip().map({"No": 0, "Yes": 1})        
        
        # Numeric Casting
        NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
        for c in NUMERIC_COLS:
            if c in df.columns:
            # Convert to numeric, replacing invalid values with NaN
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # if df["TotalCharges"].isnull().sum() !=0:
        #     df.drop(labels=df[(df['TotalCharges'].isnull()) & (df['tenure'] == 0)].index, axis=0, inplace=True)
        #     # For other rows where 'TotalCharges' is NaN (but 'tenure' is not 0), replace NaN with the mean of 'TotalCharges'
        #     mean_total_charges = df['TotalCharges'].mean()
        #     df['TotalCharges'] = df['TotalCharges'].fillna(mean_total_charges)
        #     # If there's only one row with NaN, replace it with 0
        #     if df['TotalCharges'].isnull().sum() == 1:
        #         df['TotalCharges'] = df['TotalCharges'].fillna(0)
        
        if "TotalCharges" in df.columns:
           
            if is_training:
                # --- TRAINING LOGIC (Drop rows) ---
                if "tenure" in df.columns and df["TotalCharges"].isnull().sum() !=0:
                    # Drop rows where TotalCharges is NaN AND Tenure is 0
                    drop_indices = df[(df['TotalCharges'].isnull()) & (df['tenure'] == 0)].index
                    df.drop(index=drop_indices, inplace=True)
               
                # Calculate mean on valid data & Fill remaining NaNs
                self.mean_total_charges = df['TotalCharges'].mean()
                df['TotalCharges'] = df['TotalCharges'].fillna(self.mean_total_charges)

            else:
                # --- INFERENCE LOGIC (Fill rows) ---
                if "tenure" in df.columns and "MonthlyCharges" in df.columns:
                    # Case B: Tenure != 0 -> TotalCharges = MonthlyCharges (User Requirement)
                    mask_nonzero_tenure = (df['TotalCharges'].isnull())
                    df.loc[mask_nonzero_tenure, 'TotalCharges'] = df.loc[mask_nonzero_tenure, 'MonthlyCharges']
               
                # Final safety net: If any NaNs remain (e.g. MonthlyCharges was also NaN), fill with 0
                df['TotalCharges'] = df['TotalCharges'].fillna(0)


        binary_col_list = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
        valid_columns_binary = [col for col in binary_col_list if col in df.columns]
        df[valid_columns_binary] = df[valid_columns_binary].replace({'Yes':1, 'No':0, 'Male': 1, 'Female': 0 }).astype(int)

        Multiple_col_list = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport","Contract","PaymentMethod"]
        valid_columns_multi = [col for col in Multiple_col_list if col in df.columns]

        for col in valid_columns_multi:
            if is_training:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df[col] = le.transform(df[col].astype(str))
                    
        # label_encoder = LabelEncoder()
        # for col in valid_columns_multi:
        #     df[col] = label_encoder.fit_transform(df[col])

        #convert all bool col to int
        bool_cols_list = df.select_dtypes(include='bool').columns
        df[bool_cols_list] = df[bool_cols_list].astype(int)

        # 10. SCALING (Stateful)
        cols_to_scale = [c for c in NUMERIC_COLS if c in df.columns]
        if is_training:
            print(f"DEBUG: Scaling Columns Found: {cols_to_scale}") # <--- DEBUG PRINT
            if not cols_to_scale:
                print("🚨 CRITICAL ERROR: No numeric columns found to scale! Scaler will be empty!")
            else:
                df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
        else:
            # Only transform if we have columns AND the scaler is fitted
            if cols_to_scale:
                try:
                    df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
                except Exception as e:
                    # Fallback for empty scaler (prevent crash, but warn)
                    print(f"⚠️ Scaler failed (Not Fitted?): {e}")

        # data = df.drop(['customerID'],axis=1)
        if is_training:
            # Drop target before saving schema
            if target_col in df.columns:
                self.training_features = df.drop(columns=[target_col]).columns.tolist()
            else:
                self.training_features = df.columns.tolist()
           
            # For training, we just return the df (fillna is good safety)
            return df.fillna(0)
           
        else:
            # INFERENCE: This is where reindex is MAGIC.
            # It forces the new data to have the EXACT same columns as training.
            # If a column is missing -> It adds it and fills with 0.
            if self.training_features is not None:
                # We drop target_col from input if user accidentally sent it
                if target_col in df.columns:
                    df = df.drop(columns=[target_col])
                   
                df = df.reindex(columns=self.training_features, fill_value=0)
        return df


    def training_testing_data(self, df: pd.DataFrame, target_col: str = "Churn") -> tuple:
        # Separate features (X) and target (y)
        config = Core_Operations().load_config()
        test_size = config["test_size"]
        X = df.drop(columns=[target_col])  # Drop target column
        y = df[target_col]  # The target column
        
        # Split the dataset into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=12,stratify=y)
        
        # Return the splits
        return x_train, y_train, x_test, y_test

