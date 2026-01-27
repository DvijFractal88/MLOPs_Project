# 📡 End-to-End MLOps Pipeline: Telco Customer Churn Prediction

A production-grade Machine Learning pipeline that automates the lifecycle of a churn prediction model. This project demonstrates a modular architecture handling **Data Ingestion**, **Validation**, **Training**, **Experiment Tracking (MLflow)**, and **Dockerized Deployment** on Azure.

## 🚀 Executive Summary

This system implements a robust MLOps workflow where every stage is managed by a dedicated component:
* **Data Integrity:** Validates input data schema and quality using **Great Expectations**.
* **Experimentation:** Uses **MLflow** (with a local SQLite backend) to track hyperparameters, metrics, and artifacts.
* **Reproducibility:** Models are exported and "baked" into Docker images for stable, offline deployment.
* **Scalability:** Deployed as a scalable microservice using **FastAPI** and **Azure Container Instances (ACI)**.

---

## 📂 Project Structure

```bash
├── config.yaml                    # Master Config (Hyperparameters, Paths, SQL Connection)
├── Dockerfile                     # Blueprint for building the production Docker image
├── requirements.txt               # Python dependencies
├── mlflow.db                      # SQLite Database for MLflow Tracking (Local Registry)
├── mlruns/                        # Folder storing MLflow artifacts (Models, Plots)
├── docker_model/                  # (Generated) Stores the model artifacts for the Docker container
├── deployment/                    # Configuration/Scripts specific to Cloud/Azure
└── src/
    ├── data_ingestion.py          # Step 1: Loads raw data from source (CSV/DB)
    ├── validate_data.py           # Step 2: Validates schema using Great Expectations
    ├── pre_processing.py          # Step 3: Cleaning, Encoding, and Scaling
    ├── train.py                   # Step 4: Trains XGBoost/LightGBM & logs to MLflow
    ├── export_model.py            # CI/CD: Fetches 'Production' model for Docker build
    ├── deploy.py                  # Deploy: Script to trigger deployment (Azure/ACI)
    ├── serving.py                 # Core Logic: Smart Model Loading (Online vs Offline)
    ├── app.py                     # API: FastAPI endpoints (/predict, /health)
    └── utils.py                   # Helper functions (Logging, Path management)
```

## 🛠️ Prerequisites
* OS: Windows (Development) & Linux (Production)

* Python: 3.10+

* Docker: Installed and running.

* Azure: Service Principal (SPN) credentials (for cloud deployment).



## 🚦 Execution Guide (Step-by-Step)
Follow these phases to execute the full pipeline.

### Phase 1: Data Pipeline & Training (Local)
* 1. Data Ingestion Loads the raw data from the source defined in config.yaml to your raw data folder.

```bash python src/data_ingestion.py ```
* 2. Data Validation Checks the data quality (nulls, types, ranges) using Great Expectations. If validation fails, the pipeline halts to prevent bad data from entering the model.

```bash python src/validate_data.py ```
* 3. Preprocessing Cleans the data, performs One-Hot Encoding/Scaling, and saves the preprocessor.pkl.

```bash python src/pre_processing.py ```
* 4. Model Training Trains the model (XGBoost/LightGBM) and logs metrics/artifacts to the local mlflow.db. You are not suppose to run anu of above script, as train will execute all necessary operations.

```bash python src/train.py```
* 5. Verify in MLflow Open the UI to inspect runs, compare accuracy/AUC, and view artifacts.

```bash mlflow ui --backend-store-uri sqlite:///mlflow.db ```
### Access at [http://127.0.0.1:5050](http://127.0.0.1:5050)

## Model Lifecycle (Promotion)
Register: In the MLflow UI, select your best run and click "Register Model". Name it telco_churn_model.

Promote: Transition the model version you want to deploy from Staging to Production.

## Export & Docker Generation (CI/CD Build)
To ensure stability, the Docker container does not depend on the MLflow server at runtime. We "bake" the model into the image.

* 1. Export Production Model This script queries MLflow for the specific model tagged as Production and downloads it to the docker_model/ folder.

```bash python src/export_model.py```
* 2. Build Docker Image Builds the image using the exported model files found in docker_model/, transfer all the necessary files and generate the image is possible to execute deploy.py file

* 3. Test Locally Run the container to ensure it works in "Offline Mode".

Bash

docker run -d -p 8100:8100 --name my_churn_model telco-churn-api
Check Health: Visit http://localhost:8100/docs to see the Swagger UI.

Transfer to Linux (Production)
To deploy on a remote Linux server:

Transfer Files: Copy the src/ folder, Dockerfile, requirements.txt, and config.yaml to the server.

Transfer Model: Copy the generated docker_model/ folder from your dev machine to the Linux server.

Build & Run: Run the standard docker build and docker run commands on the Linux terminal.


## 🔌 API Usage
Endpoint: POST /predict

Sample Request:

JSON

{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
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
  "TotalCharges": 340.5
}
Sample Response:

JSON

{
  "prediction": 0,
  "churn_probability": 0.142
}

### Use this using swagger UI means http://localhost:8100/docs

### steps to execute:
# Model Training and Deployment Workflow

## 1. Upload the Data
- Place the raw data in the `data/raw` folder.

## 2. Train the Model
- Run the `train.py` file to:
  - Perform data preprocessing
  - Validate the data
  - Tune hyperparameters
  - Train the model
- Track the entire training process using **MLflow** locally (due to conflicts with other users).

## 3. Model Prediction and Serving
- Use the `serving.py` file to check the predictions from the trained model.
- Select the best model based on performance metrics.

## 4. Export the Best Model
- Export the model with the highest **AUC-ROC** score.
- Manually change the model stage from `None` → `Staging` → `Production` → `Archive`.  
  *(Note: The code for this stage has been removed but can be found in the Git version history.)*

## 5. Export Model for Docker
- The `export_model.py` file will:
  - Generate the `docker_model/` folder.
  - Only the best model will be moved to the folder.
- Transfer the model to the **Linux server**, as Docker is hosted there.

## 6. Docker Image Creation and Deployment
- Execute the `deploy.py` file to:
  - Automatically generate the Docker image.
  - Test the model on the Linux server at IP `8100`.

## Link of the document which has screenshots: (https://docs.google.com/document/d/1baWKBpRYAg-5dSdn7buIw7OyDmt63iWFPRvV8H36WW4/edit?usp=sharing)