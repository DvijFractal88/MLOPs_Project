import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import pandas as pd
import joblib
import os
import yaml
from mlflow.tracking import MlflowClient

class ModelService:
    def __init__(self):
        # 1. Load Config
        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
           
        mlflow.set_tracking_uri(self.config['ml_tracing_url'])
        self.client = MlflowClient()
        self.model_name = self.config['model_name']
       
        print(f"--- Initializing Service for {self.model_name} ---")
        self.load_model_and_artifacts()

    def load_model_and_artifacts(self):
        # A. Find the Latest Version (Preferably Production)
        try:
            latest_version = self.client.get_latest_versions(self.model_name, stages=["Production"])[0]
            print(f"✅ Found PRODUCTION model: Version {latest_version.version}")
        except IndexError:
            latest_version = self.client.get_latest_versions(self.model_name, stages=["None"])[0]
            print(f"⚠️ Production model not found. Using LATEST version: {latest_version.version}")

        run_id = latest_version.run_id
        model_uri = f"runs:/{run_id}/model"
       
        # B. Identify the Algorithm to use the Correct Loader
        # We fetch the run tags to see which algorithm won (xgboost, lightgbm, etc.)
        run = self.client.get_run(run_id)
        model_algo = run.data.tags.get("model_algorithm", "sklearn") # Default to sklearn if missing
       
        print(f"🔍 Algorithm Detected: {model_algo.upper()}")
        print(f"📥 Loading Native Model from: {model_uri}")
       
        # C. Load the NATIVE Model (Preserves .predict_proba)
        if model_algo == 'xgboost':
            self.model = mlflow.xgboost.load_model(model_uri)
        elif model_algo == 'lightgbm':
            self.model = mlflow.lightgbm.load_model(model_uri)
        else:
            # Random Forest & Logistic Regression are handled by sklearn
            self.model = mlflow.sklearn.load_model(model_uri)
       
        # D. Download & Load the Preprocessor
        print("📥 Downloading Preprocessor...")
        if not os.path.exists("preprocessor.pkl"):
            self.client.download_artifacts(run_id, "data/preprocessor.pkl", dst_path=".")
       
        # Load local file
        self.preprocessor = joblib.load("preprocessor.pkl")
        print("✅ Preprocessor Loaded Successfully.")

    def predict(self, input_data: dict):
        # 1. Convert Dict to DataFrame
        df = pd.DataFrame([input_data])
       
        # 2. Preprocess (Force is_training=False)
        processed_df = self.preprocessor.preprocess_data(df, is_training=False)
       
        # 3. Predict (Native Model supports predict_proba now)
        prediction = self.model.predict(processed_df)
        print("Predication is done=====>",prediction)
        # Handle different output formats for probability
        try:
            # Most sklearn/xgb models return [prob_class_0, prob_class_1]
            probability = self.model.predict_proba(processed_df)[:, 1]
        except:
            # Fallback if predict_proba isn't available
            probability = [0.0]
       
        return {
            "prediction": int(prediction[0]),
            "churn_probability": float(probability[0])
        }

# --- Quick Test Block ---
if __name__ == "__main__":
    service = ModelService()
   
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