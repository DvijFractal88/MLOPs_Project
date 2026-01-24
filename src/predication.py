import pandas as pd
import mlflow
import json
import numpy as np

# Import your local helper modules
# Ensure validate_data.py and pre_processing.py are in the same folder
from validate_data import validate_telco_data
from pre_processing import DataPreProcessing

class ChurnPredictor:
    def __init__(self, model_name="TelcoChurnModel-DB-RF", stage="Production"):
        """
        Initializes the predictor by loading the Production model from MLflow.
        """
        self.model_name = model_name
        self.stage = stage
        self.model = self._load_model()
       
        # Initialize your preprocessor class
        self.preprocessor = DataPreProcessing()

    def _load_model(self):
        """Internal helper to load model from MLflow Registry"""
        try:
            # Point this to where your MLflow data is stored
            # If running locally/Docker with file storage:
            mlflow.set_tracking_uri("http://127.0.0.1:5000/")
           
            model_uri = f"models:/{self.model_name}/{self.stage}"
            print(f"--- Loading '{self.stage}' model from: {model_uri} ---")
           
            # This 'pyfunc' loader is generic. It works for XGBoost, Sklearn, and TF.
            model = mlflow.pyfunc.load_model(model_uri)
            print("Successfully loaded model.")
            return model
           
        except Exception as e:
            print(f"CRITICAL ERROR: Could not load model. Details: {e}")
            print("Did you run the training script and promote a model to Production yet?")
            return None

    def predict(self, json_input):
        """
        The Main Function:
        1. Accepts JSON
        2. Validates Data Quality
        3. Preprocesses (Scaling/Encoding)
        4. Predicts Churn
        """
        # --- 1. PARSE JSON INPUT ---
        try:
            # Handle if input is string or python dict/list
            if isinstance(json_input, str):
                data = json.loads(json_input)
            else:
                data = json_input
           
            # Ensure it is a list of records
            if isinstance(data, dict):
                data = [data]
               
            raw_df = pd.DataFrame(data)
           
        except Exception as e:
            return {"status": "error", "message": f"Invalid JSON format: {str(e)}"}

        # --- 2. VALIDATE DATA ---
        # This uses the great_expectations code we wrote earlier
        print(f"Validating {len(raw_df)} records...")
        is_valid, failed_issues = validate_telco_data(raw_df)
       
        if not is_valid:
            error_msg = f"Data Validation Failed. Issues: {failed_issues}"
            print(f"❌ {error_msg}")
            return {
                "status": "failure",
                "reason": "validation_error",
                "details": failed_issues
            }

        # --- 3. PREPROCESS DATA ---
        print("Preprocessing data...")
        try:
            # We assume your DataPreProcessing class has a method that takes raw DF
            # and returns the formatted X features ready for the model.
            # IMPORTANT: This must match training logic exactly (filling NaNs, encoding).
            processed_data = self.preprocessor.preprocess_data(raw_df)
           
            # If your preprocessor returns tuple (X, y), grab just X
            if isinstance(processed_data, tuple):
                processed_data = processed_data[0]
               
        except Exception as e:
            return {"status": "error", "message": f"Preprocessing failed: {str(e)}"}

        # --- 4. RUN PREDICTION ---
        print("Running inference...")
        try:
            if self.model is None:
                return {"status": "error", "message": "Model not initialized"}
            print("Processed_data ====> \n",processed_data)
            # Get predictions
            predictions = self.model.predict(processed_data)
           
            # Format the output nicely
            results = []
           
            # Loop through results to format them
            # (predictions might be a numpy array or list)
            for i, pred in enumerate(predictions):
                # Handle different model output types (Float vs Int vs Array)
                if isinstance(pred, (list, np.ndarray)) and len(pred) == 1:
                    score = float(pred[0])
                else:
                    score = float(pred)
               
                # Logic: If > 0.5, it is Churn
                # Note: If you optimized a custom threshold (e.g. 0.35), apply it here
                churn_label = "Yes" if score > 0.5 else "No"
               
                record_id = raw_df.iloc[i].get("customerID", f"Row_{i}")
               
                results.append({
                    "customerID": record_id,
                    "churn_prediction": churn_label,
                    "churn_probability": round(score, 4)
                })
               
            return {"status": "success", "data": results}

        except Exception as e:
            return {"status": "error", "message": f"Prediction runtime error: {str(e)}"}

# ==========================================
# TEST SECTION (Run this file directly to test)
# ==========================================
if __name__ == "__main__":
   
    # Initialize Service
    service = ChurnPredictor()
   
    # Example JSON Payload (Matches your dataset structure)
    test_payload = [
        {
            "customerID": "TEST-USER-01",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "Yes",
            "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV":"Yes",
            "StreamingMovies":"No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85,
            "TotalCharges": 29.85
        }
    ]
   
    print("\n--- Sending Test Request ---")
    response = service.predict(test_payload)
   
    print("\n--- Response ---")
    print(json.dumps(response, indent=2))