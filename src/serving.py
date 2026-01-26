import mlflow
import pandas as pd
import joblib
import os
import yaml
from mlflow.tracking import MlflowClient
from utils import Core_Operations
from data_ingestion import DataIngestion
from pre_processing import DataPreProcessing

class ModelService:
    def __init__(self):
        # 1. Load Config to get URI and Model Name
        self.config = Core_Operations().load_config()
           
        mlflow.set_tracking_uri(self.config['ml_tracing_url'])
        self.client = MlflowClient()
        self.model_name = self.config['model_name']
       
        print(f"--- Initializing Service for {self.model_name} ---")
        self.load_model_and_artifacts()

    def load_model_and_artifacts(self):
        """
        Loads the 'Production' model and its attached preprocessor from MLflow.
        If no Production model exists, it falls back to the latest 'None' version.
        """
        # A. Find the Latest Version (Preferably Production)
        try:
            # Try getting the Production model first
            latest_version = self.client.get_latest_versions(self.model_name, stages=["Production"])[0]
            print(f"Found PRODUCTION model: Version {latest_version.version}")
        except IndexError:
            # Fallback: Get the latest registered version (Stage: None)
            latest_version = self.client.get_latest_versions(self.model_name, stages=["None"])[0]
            print(f"Production model not found. Using LATEST version: {latest_version.version}")

        run_id = latest_version.run_id
        model_uri = f"runs:/{run_id}/model"
       
        # B. Load the Main Model (XGB/LGBM/RF/LogReg)
        print(f"Loading model from: {model_uri}")
        self.model = mlflow.pyfunc.load_model(model_uri)
       
        # C. Download & Load the Preprocessor (From 'extras' folder)
        print("Downloading Preprocessor...")
        local_path = self.client.download_artifacts(run_id, "data/preprocessor.pkl", dst_path=".")
        self.preprocessor = joblib.load(local_path)
        print("✅ Preprocessor Loaded Successfully.")
       
        # Cleanup downloaded file to keep container clean
        # os.remove(local_path) # Optional: Enable in Docker

    def predict(self, input_data: dict):
        """
        Accepts a dictionary (single row of data), preprocesses it, and predicts.
        """
        # 1. Convert Dict to DataFrame
        df = pd.DataFrame([input_data])
        
        if "OnlineSecurity" in self.preprocessor.label_encoders:
            print("encoder found")
        else:
            print("Encoder missing")
       
        # 2. Preprocess (Force is_training=False)
        # This uses the FROZEN saved logic (Mean imputation, Scaling, Encodings)
        processed_df = self.preprocessor.preprocess_data(df, is_training=False)
       
        # 3. Predict
        # We use .predict() from the loaded MLflow model
        prediction = self.model.predict(processed_df)
        probability = self.model.predict_proba(processed_df)[:, 1]
       
        return {
            "prediction": int(prediction[0]),
            "churn_probability": float(probability[0])
        }

# --- Quick Test Block (Run this file directly to test) ---
if __name__ == "__main__":
    service = ModelService()
   
    # Fake data to test
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }
   
    result = service.predict(sample_customer)
    print("\n🔮 Prediction Result:")
    print(result)