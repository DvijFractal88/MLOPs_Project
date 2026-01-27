import mlflow
import shutil
import os
import joblib
import yaml
from mlflow.tracking import MlflowClient

# Load Config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

mlflow.set_tracking_uri(config['ml_tracing_url'])
client = MlflowClient()
model_name = config['model_name']

print(f"--- Exporting Model: {model_name} ---")

# 2. Smart Model Search (Production -> Staging -> Latest)
run_id = None
target_stage = "Unknown"

# Try Production
try:
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    if prod_versions:
        run_id = prod_versions[0].run_id
        target_stage = "Production"
except:
    pass

# Try Staging (if no Production)
if not run_id:
    try:
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
        if staging_versions:
            run_id = staging_versions[0].run_id
            target_stage = "Staging"
    except:
        pass

# Try Latest/None (if no Production or Staging)
if not run_id:
    try:
        # Get ALL versions and pick the most recent one
        all_versions = client.get_latest_versions(model_name, stages=["None"])
        if all_versions:
            run_id = all_versions[0].run_id
            target_stage = "Latest (None)"
    except Exception as e:
        print(f"CRITICAL: Could not find ANY model versions. Did you run train.py? Error: {e}")
        exit()

if not run_id:
    print("Error: No run_id found. Please check your MLflow experiments.")
    exit()

print(f"Found Target Model: {target_stage} (Run ID: {run_id})")

# 3. Prepare Clean Export Folder
output_dir = "docker_model"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# 4. Download Artifacts
print("Downloading Model & Preprocessor...")
try:
    # Download the model folder
    client.download_artifacts(run_id, "model", dst_path=output_dir)
   
    # Download the preprocessor
    client.download_artifacts(run_id, "data/preprocessor.pkl", dst_path=output_dir)
   
    # Organize files (Move preprocessor to root of docker_model)
    # Note: MLflow downloads to docker_model/data/preprocessor.pkl, we want it at docker_model/preprocessor.pkl
    source_prep = os.path.join(output_dir, "data", "preprocessor.pkl")
    dest_prep = os.path.join(output_dir, "preprocessor.pkl")
   
    if os.path.exists(source_prep):
        shutil.move(source_prep, dest_prep)
        shutil.rmtree(os.path.join(output_dir, "data")) # Clean up empty folder
       
    print(f"✅ Export Successful! Files are in '{output_dir}/'")
    print("   - You can now copy the 'docker_model' folder to Linux.")

except Exception as e:
    print(f"Export Failed: {e}")
    print("   - Check if 'preprocessor.pkl' exists in the artifacts for this run.")