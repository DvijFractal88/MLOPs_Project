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
        current_file_path = os.path.abspath(__file__)
        src_dir = os.path.dirname(current_file_path)
        self.project_root = os.path.dirname(src_dir)
        # Define paths relative to Root
        config_path = os.path.join(self.project_root, "config.yaml")
        self.docker_model_path = os.path.join(self.project_root, "docker_model")

        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        
        # Check if the "Offline Docker Model" exists
        offline_prep_path = os.path.join(self.docker_model_path, "preprocessor.pkl")
        self.model_name = self.config['model_name']
        if os.path.exists(offline_prep_path):
            print(f"Docker/Offline Mode Detected.- Loading from: {self.docker_model_path}")
            self.is_offline = True
            self.load_model_and_artifacts()
        else:
            print("Dev/Online Mode Detected. - Connecting to MLflow Server...")
            self.is_offline = False
            # --- YOUR ORIGINAL MLFLOW SETUP ---
            mlflow.set_tracking_uri(self.config['ml_tracing_url'])
            self.client = MlflowClient()
            self.load_model_and_artifacts()
       
        print(f"--- Initializing Service for {self.model_name} ---")
        # self.load_model_and_artifacts()

    def load_model_and_artifacts(self):
        # =========================================================
        # PATH A: OFFLINE MODE (Docker / Production)
        # =========================================================
        if self.is_offline:
            try:
                # 1. Load Preprocessor
                prep_path = os.path.join(self.docker_model_path, "preprocessor.pkl")
                self.preprocessor = joblib.load(prep_path)
               
                # 2. Load Native Model (Try all flavors since tags aren't available offline)
                model_dir = os.path.join(self.docker_model_path, "model")
               
                try:
                    self.model = mlflow.xgboost.load_model(model_dir)
                    print("✅ Loaded Native XGBoost Model (Offline)")
                except:
                    try:
                        self.model = mlflow.lightgbm.load_model(model_dir)
                        print("✅ Loaded Native LightGBM Model (Offline)")
                    except:
                        self.model = mlflow.sklearn.load_model(model_dir)
                        print("✅ Loaded Native Sklearn Model (Offline)")
                       
                print("✅ Service Ready (Offline Mode).")
            except Exception as e:
                print(f"Critical Error loading Offline Model: {e}")
            return # Stop here, do not run Online logic

        # =========================================================
        # PATH B: ONLINE MODE (Dev / MLflow) - YOUR ORIGINAL CODE
        # =========================================================
       
        # A. Find the Latest Version (Preferably Production)
        try:
            latest_version = self.client.get_latest_versions(self.model_name, stages=["Production"])[0]
            print(f" Found PRODUCTION model: Version {latest_version.version}")
        except IndexError:
            latest_version = self.client.get_latest_versions(self.model_name, stages=["None"])[0]
            print(f" Production model not found. Using LATEST version: {latest_version.version}")

        run_id = latest_version.run_id
        model_uri = f"runs:/{run_id}/model"
       
        # B. Identify the Algorithm
        run = self.client.get_run(run_id)
        model_algo = run.data.tags.get("model_algorithm", "sklearn") # Default to sklearn
       
        print(f" Algorithm Detected: {model_algo.upper()}")
        print(f" Loading Native Model from: {model_uri}")
       
        # C. Load the NATIVE Model
        if model_algo == 'xgboost':
            self.model = mlflow.xgboost.load_model(model_uri)
        elif model_algo == 'lightgbm':
            self.model = mlflow.lightgbm.load_model(model_uri)
        else:
            self.model = mlflow.sklearn.load_model(model_uri)
       
        # D. Download & Load the Preprocessor
        print(" Downloading Preprocessor...")
       
        # Logic: If we are in Dev, download to the local root
        local_prep_path = os.path.join(self.project_root, "preprocessor.pkl")
       
        if not os.path.exists(local_prep_path):
            self.client.download_artifacts(run_id, "data/preprocessor.pkl", dst_path=self.project_root)
            # Move it from project_root/data/preprocessor.pkl if MLflow created a subfolder
            possible_subfolder = os.path.join(self.project_root, "data", "preprocessor.pkl")
            if os.path.exists(possible_subfolder):
                 os.replace(possible_subfolder, local_prep_path)
                 try: os.rmdir(os.path.join(self.project_root, "data"))
                 except: pass

        self.preprocessor = joblib.load(local_prep_path)
        print(" Preprocessor Loaded Successfully.")

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