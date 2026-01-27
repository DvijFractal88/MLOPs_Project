from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from serving import ModelService
from train import ChurnModelPipeline
import uvicorn
import pandas as pd

# 1. Initialize FastAPI
app = FastAPI(title="Telco Churn Prediction API", version="2.0")

# 2. Global Model Service Loading
print("Initializing Model Service...")
try:
    model_service = ModelService()
    print("Model Service Ready!")
except Exception as e:
    print(f"Warning: Could not load model on startup. Train a model first. Error: {e}")
    model_service = None

# 3. Custom Error Handling for Clear Validation Messages
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"status": "error", "message": "Invalid Data Format", "details": exc.errors()},
    )

# 4. Input Schema (Pydantic enforces types & validation)
class CustomerData(BaseModel):
    gender: str = Field(..., pattern="^(Male|Female)$")
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str = Field(..., pattern="^(Yes|No)$")
    Dependents: str = Field(..., pattern="^(Yes|No)$")
    tenure: int = Field(..., ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(..., gt=0)
    TotalCharges: float = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
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
                "TotalCharges": 358.20
            }
        }

# --- ENDPOINT 1: HEALTH ---
@app.get("/health")
def health_check():
    """
    Simple GET request to check if API is alive.
    """
    model_status = "loaded" if model_service else "not_loaded"
    return {"status": "ok", "model_status": model_status}

# --- ENDPOINT 2: PREDICT ---
@app.post("/predict")
def predict(customer: CustomerData):
    """
    Accepts customer JSON, validates it, and returns Churn Probability.
    """
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please train a model first.")

    try:
        # Convert Pydantic object to dictionary
        input_data = customer.model_dump()
       
        # Call the Serving Logic (handles pre-processing & probability internally)
        result = model_service.predict(input_data)
       
        return {
            "status": "success",
            "prediction": result["prediction"],
            "probability": result["churn_probability"],
            "risk_level": "High" if result["churn_probability"] > 0.7 else "Low"
        }

    except Exception as e:
        # Catch unexpected errors (like pre-processing crashes)
        raise HTTPException(status_code=500, detail=f"Prediction Failed: {str(e)}")

# --- ENDPOINT 3: TRAIN ---
def run_training_task():
    """Background task to run training"""
    print("Background Training Started...")
    try:
        pipeline = ChurnModelPipeline()
        pipeline.run_pipeline()
       
        # Reload the service with new model
        global model_service
        model_service = ModelService()
        print("Training Complete & Service Reloaded.")
    except Exception as e:
        print(f"Training Failed: {e}")

@app.post("/train")
def trigger_training(background_tasks: BackgroundTasks):
    """
    Triggers the training pipeline in background.
    Returns immediately.
    """
    background_tasks.add_task(run_training_task)
    return {"status": "ok", "message": "Training started in background."}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8100, reload=True)