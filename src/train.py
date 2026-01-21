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
import mlflow.sklearn
import mlflow.xgboost
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

class train_model:

    def data_to_train_model(self):
        raw_data = DataIngestion().load_raw_csv_file()
        is_valid, failed = validate_telco_data(raw_data)
       
        processed_data = DataPreProcessing().preprocess_data(raw_data)
        Core_Operations().save_to_csv_file(processed_data)
        data = DataIngestion().load_processed_csv_file()
        x_train, y_train, x_test, y_test = DataPreProcessing().training_testing_data(data)
        print(f"training data size : {x_train.shape}")
        print(f"test data size : {x_test.shape}")
        return x_train, y_train, x_test, y_test,is_valid,failed,data
        
    
    def data_for_testing(self,input_json):
        print("Will do later")
    
    def train_model(self): #Json_argument
        X_train, y_train, X_test, y_test,is_valid,failed,data = self.data_to_train_model()
        if not is_valid:
            # Log validation failures for debugging
            import json
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
            raise ValueError(f"❌ Data quality check failed. Issues: {failed}")
        else:
            print("✅ Data validation passed. Logged to MLflow.")
        model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
        )
        mlflow.set_tracking_uri("http://127.0.0.1:5000/")
        experiment_name = "ml_test_dvijesh"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            # If not found, create it
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id

        # Set the experiment context explicitly
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(experiment_id=experiment_id):
            # Train model
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            rec = recall_score(y_test, preds)

            # Log params, metrics, and model
            mlflow.log_param("n_estimators", 350)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("recall", rec)
            mlflow.xgboost.log_model(model, "model")

            # 🔑 Log dataset so it shows in MLflow UI
            train_ds = mlflow.data.from_pandas(data, source="training_data")
            mlflow.log_input(train_ds, context="training")

            print(f"Model trained. Accuracy: {acc:.4f}, Recall: {rec:.4f}") 
        
    
def main():
    model = train_model()  # Create an instance of the class
    model.train_model()  # Call the train_model method

if __name__ == "__main__":
    main()