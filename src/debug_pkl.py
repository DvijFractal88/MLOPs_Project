import joblib
import pandas as pd
import os

# 1. Define the path to your LOCAL file
# (This must be the file created by your latest train.py run)
local_pkl_path = "mlruns/1/24e1f77691a846c1ae62e7c33828853d/artifacts/data/preprocessor.pkl"

if not os.path.exists(local_pkl_path):
    print(f"❌ ERROR: File not found at {local_pkl_path}")
    print("   Please run 'python src/train.py' first to generate it.")
    exit()

print(f"✅ Found local preprocessor at: {local_pkl_path}")

# 2. Load it DIRECTLY (Bypassing MLflow)
try:
    preprocessor = joblib.load(local_pkl_path)
    print("✅ Preprocessor loaded successfully.")
except Exception as e:
    print(f"❌ CRITICAL: Could not load pkl file. Corrupted? Error: {e}")
    exit()

# 3. Inspect the Internals (The "X-Ray")
print("\n--- INSPECTING INTERNAL STATE ---")

# Check Scaler
if hasattr(preprocessor, 'scaler'):
    try:
        # If mean_ is present, it means .fit() worked!
        print(f"Scaler Mean: {preprocessor.scaler.mean_}")
        print("✅ Scaler is FITTED.")
    except AttributeError:
        print("❌ CRITICAL: Scaler is NOT FITTED. (mean_ attribute is missing)")
else:
    print("❌ Scaler object is missing entirely.")

# Check Encoders
if hasattr(preprocessor, 'label_encoders'):
    print(f"Encoders Found: {list(preprocessor.label_encoders.keys())}")
    if len(preprocessor.label_encoders) > 0:
        print("✅ Label Encoders are present.")
    else:
        print("❌ CRITICAL: Label Encoders dict is empty.")
else:
    print("❌ label_encoders attribute is missing.")

print("\n---------------------------------")

# 4. Test Run (Simulate Serving)
sample_customer = pd.DataFrame([{
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
}])

try:
    print("Attempting to transform data...")
    # Force is_training=False to test inference logic
    result = preprocessor.preprocess_data(sample_customer, is_training=False)
    print("✅ Transformation Successful!")
    print(result.head())
except Exception as e:
    print(f"❌ Transformation FAILED: {e}")
