import os
import optuna
import mlflow
from xgboost import XGBClassifier
from data_ingestion import DataIngestion
from pre_processing import DataPreProcessing
from utils import Core_Operations
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, roc_auc_score
)
from validate_data import validate_telco_data

class train_model:

    def data_to_train_model(self):
        raw_data = DataIngestion().load_raw_csv_file()
        is_valid, failed = validate_telco_data(raw_data)
        if not is_valid:
            # Log validation failures for debugging
            import json
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
            raise ValueError(f"❌ Data quality check failed. Issues: {failed}")
        else:
            print("✅ Data validation passed. Logged to MLflow.") 
        processed_data = DataPreProcessing().preprocess_data(raw_data)
        Core_Operations().save_to_csv_file(processed_data)
        data = DataIngestion().load_processed_csv_file()
        x_train, y_train, x_test, y_test = DataPreProcessing().training_testing_data(data)
        print(f"training data size : {x_train.shape}")
        print(f"test data size : {x_test.shape}")
        return x_train, y_train, x_test, y_test
        
    
    def data_for_testing(self,input_json):
        print("Will do later")
    
    def train_model(self, Json_argument):
        print("traning model")
    
def main():
    model = train_model()  # Create an instance of the class
    model.data_to_train_model()  # Call the train_model method

if __name__ == "__main__":
    main()