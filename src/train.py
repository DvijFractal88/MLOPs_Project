import os
import optuna
import mlflow
from xgboost import XGBClassifier
from data_ingestion import DataIngestion
from pre_processing import DataPreProcessing
from utils import Core_Operations
from mlflow.models.signature import infer_signature
import shutil
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    classification_report, accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score
)
from validate_data import validate_telco_data
import mlflow.sklearn
import mlflow.xgboost
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

class ChurnModelPipeline:

    def __init__(self, experiment_name="telco_churn_optimization_DB"):
        mlflow.set_tracking_uri("http://127.0.0.1:5000/")
        self.experiment_name = experiment_name
        self.model_name = "TelcoChurnModel-DB"  # Define the Registry Name
       
        # Ensure experiment exists
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
           
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()

    def load_and_validate_data(self):
        raw_data = DataIngestion().load_raw_csv_file()
        is_valid, failed = validate_telco_data(raw_data)
        if not is_valid:
            raise ValueError(f"Data validation failed: {failed}")
        
        processed_data = DataPreProcessing().preprocess_data(raw_data)
        Core_Operations().save_to_csv_file(processed_data)
        data = DataIngestion().load_processed_csv_file()
        x_train, y_train, x_test, y_test = DataPreProcessing().training_testing_data(data)

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        return x_train, y_train, x_test, y_test, data, scale_pos_weight

    def calculate_metrics(self, y_true, y_pred, y_probs):
        return{
            "accuracy": accuracy_score(y_true,y_pred),
            "recall" : recall_score(y_true, y_pred),
            "precision": precision_score(y_true,y_pred),
            "f1": f1_score(y_true,y_pred),
            "roc_auc": roc_auc_score(y_true,y_pred)
        }

    def objective(self, trial, X_train, y_train, X_test, y_test, base_scale_pos_weight):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.01),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', base_scale_pos_weight * 0.5, base_scale_pos_weight * 1.2),
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1
        }

        with mlflow.start_run(nested=True, run_name=f"Trial_{trial.number}"):
            model = XGBClassifier(**param)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1] # Probability for ROC_AUC
            # Calculate ALL metrics
            metrics = self.calculate_metrics(y_test, preds, probs)

             # Log Params & Metrics to MLflow
            mlflow.log_params(param)
            mlflow.log_metrics(metrics)

            return metrics["roc_auc"]
    
    def run_pipeline(self, n_trials=10):
        X_train, y_train, X_test, y_test, full_data, base_weight = self.load_and_validate_data()
       
        print(f"--- 1. Optimizing for ROC_AUC (Logging all 5 metrics) ---")
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_test, y_test, base_weight),
            n_trials=n_trials
        )
       
        print(f"--- 2. Training Champion Model ---")
        print(f"Best Params: {study.best_params}")
        self.train_and_register(study.best_params, X_train, y_train, X_test, y_test, full_data)

    def train_and_register(self, params, X_train, y_train, X_test, y_test, full_data):
        with mlflow.start_run(run_name="Champion_Model_Candidate") as run:
            model = XGBClassifier(**params)
            model.fit(X_train, y_train)
           
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]
           
            # Calculate & Log ALL Metrics for the Champion
            metrics = self.calculate_metrics(y_test, preds, probs)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
           
            # Log Data & Model
            train_ds = mlflow.data.from_pandas(full_data, source="processed_telco_data")
            mlflow.log_input(train_ds, context="training")
           
            signature = infer_signature(X_test, preds)
            model_info = mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path="model",
                signature=signature,
                registered_model_name=self.model_name
            )
           
            print(f"Candidate Model Registered. Metrics: {metrics}")
           
            # Pass the ROC_AUC to the promotion logic
            self.evaluate_and_promote(
                new_version=model_info.registered_model_version,
                new_metrics=metrics
            )

    def data_for_testing(self,input_json):
        print("Will do later")
    
    def evaluate_and_promote(self, new_version, new_metrics):
        print("--- Evaluating for Promotion ---")
       
        # Primary Metric for Decision Making
        DECISION_METRIC = "roc_auc"
        new_score = new_metrics[DECISION_METRIC]

        try:
            prod_models = self.client.get_latest_versions(self.model_name, stages=["Production"])
        except:
            prod_models = []

        if not prod_models:
            print("No Production model found. Promoting immediately.")
            self.promote_to_production(new_version)
            return

        current_prod = prod_models[0]
        prod_run = self.client.get_run(current_prod.run_id)
        prod_score = prod_run.data.metrics.get(DECISION_METRIC, 0.0)

        print(f"Comparing {DECISION_METRIC}:")
        print(f"New Model (v{new_version}): {new_score:.4f}")
        print(f"Production (v{current_prod.version}): {prod_score:.4f}")

        # Strict Improvement Check
        if new_score > prod_score:
            print(">>> IMPROVEMENT DETECTED. Promoting.")
            self.promote_to_production(new_version)
        else:
            print(f">>> NO IMPROVEMENT in {DECISION_METRIC}. Keeping in Staging.")
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=new_version,
                stage="Staging"
            )

    def promote_to_production(self, version):
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Success! Version {version} is now Production.")
    
    

    def export_model(self):
        MODEL_NAME = "TelcoChurnModel-DB"
        STAGE = "Production"
        EXPORT_FOLDER = "./my_production_model"
        ZIP_FILENAME = "production_model_pack"
        print(f"--- Looking for '{STAGE}' model: {MODEL_NAME} ---")

        # 1. Clean up old folders if they exist
        if os.path.exists(EXPORT_FOLDER):
            shutil.rmtree(EXPORT_FOLDER)
        if os.path.exists(f"{ZIP_FILENAME}.zip"):
            os.remove(f"{ZIP_FILENAME}.zip")

        try:
            # 2. THE MAGIC PART
            # This one line finds the correct folder inside mlruns and copies it to EXPORT_FOLDER
            model_uri = f"models:/{MODEL_NAME}/{STAGE}"
        
            print(f"Downloading artifacts from: {model_uri} ...")
            mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=EXPORT_FOLDER)
        
            print("Success! Model files downloaded.")
        
            # 3. Zip it up
            shutil.make_archive(ZIP_FILENAME, 'zip', EXPORT_FOLDER)
            print(f"\nCREATED ZIP FILE: {os.path.abspath(ZIP_FILENAME + '.zip')}")
            print("You can now send this zip file to your Linux machine.")

        except Exception as e:
            print("\n[ERROR] Could not find the model.")
            print("Did you promote a model to Production yet?")
            print(f"Error details: {e}")
        
    
  # Call the train_model method

if __name__ == "__main__":
    pipeline = ChurnModelPipeline()
    # pipeline.run_pipeline(n_trials=10)
    pipeline.export_model()