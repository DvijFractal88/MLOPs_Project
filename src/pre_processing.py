import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import Core_Operations


class DataPreProcessing:

    def preprocess_data(self, df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
        df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace
        
        binary_col_list = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
        NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
        Multiple_col_list = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport","StreamingTV","StreamingMovies","Contract","PaymentMethod"]
        
         # drop ids if present
        for col in ["customerID", "CustomerID", "customer_id"]:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # target to 0/1 if it's Yes/No
        if target_col in df.columns and df[target_col].dtype == "object":
            df[target_col] = df[target_col].str.strip().map({"No": 0, "Yes": 1})
        
        # TotalCharges often has blanks in this dataset -> coerce to float
        for c in NUMERIC_COLS:
            if c in df.columns:
            # Convert to numeric, replacing invalid values with NaN
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # if "TotalCharges" in df.columns:
        #     df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        
        
        valid_columns_binary = [col for col in binary_col_list if col in df.columns]
        df[valid_columns_binary] = df[valid_columns_binary].replace({'Yes':1, 'No':0, 'Male': 1, 'Female': 0 }).astype(int)

        
        valid_columns_multi = [col for col in Multiple_col_list if col in df.columns]
        df = pd.get_dummies(df,columns=valid_columns_multi, drop_first= True)

        #convert all bool col to int
        bool_cols_list = df.select_dtypes(include='bool').columns
        df[bool_cols_list] = df[bool_cols_list].astype(int)

        if df["TotalCharges"].isnull().sum() !=0:
            df.drop(labels=df[(df['TotalCharges'].isnull()) & (df['tenure'] == 0)].index, axis=0, inplace=True)
            # For other rows where 'TotalCharges' is NaN (but 'tenure' is not 0), replace NaN with the mean of 'TotalCharges'
            mean_total_charges = df['TotalCharges'].mean()
            df['TotalCharges'] = df['TotalCharges'].fillna(mean_total_charges)
            # If there's only one row with NaN, replace it with 0
            if df['TotalCharges'].isnull().sum() == 1:
                df['TotalCharges'] = df['TotalCharges'].fillna(0)

        # data = df.drop(['customerID'],axis=1)
        data = df.reindex(columns=df.columns, fill_value=0)
        return data


    def training_testing_data(self, df: pd.DataFrame, target_col: str = "Churn") -> tuple:
        # Separate features (X) and target (y)
        config = Core_Operations().load_config()
        test_size = config["test_size"]
        X = df.drop(columns=[target_col])  # Drop target column
        y = df[target_col]  # The target column
        
        # Split the dataset into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Return the splits
        return x_train, y_train, x_test, y_test

