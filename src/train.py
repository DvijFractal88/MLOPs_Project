import os
import optuna
import mlflow
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from data_ingestion import DataIngestion
from pre_processing import DataPreProcessing
from utils import Core_Operations
from mlflow.models.signature import infer_signature
import shutil
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    classification_report, accuracy_score, recall_score, precision_score,f1_score, roc_auc_score, confusion_matrix
)
from validate_data import validate_telco_data
import mlflow.sklearn
import mlflow.xgboost
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


config = Core_Operations().load_config()

class ChurnModelPipeline:

    def __init__(self):
        mlflow.set_tracking_uri(config["ml_tracing_url"])
        self.experiment_name= config["mlflow_experiment_name"]
        self.model_name = config["model_name"] #"TelcoChurnModel-DB-LGB"  # Define the Registry Name
        self.current_model_name = None
        # Ensure experiment exists
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
           
        mlflow.set_experiment(self.experiment_name)
        self.scale_pos_weight = 0.0
        self.client = MlflowClient()
        self.temp_artifact_dir = "temp_model_artifacts"
        if os.path.exists(self.temp_artifact_dir):
            shutil.rmtree(self.temp_artifact_dir)
        os.makedirs(self.temp_artifact_dir)

    def load_and_validate_data(self):
        raw_data = DataIngestion().load_raw_csv_file()
        is_valid, failed = validate_telco_data(raw_data)
        if not is_valid:
            raise ValueError(f"Data validation failed: {failed}")
        preprocessor = DataPreProcessing()
        processed_data = preprocessor.preprocess_data(raw_data)
        Core_Operations().save_to_csv_file(processed_data)
        # data = DataIngestion().load_processed_csv_file()
        
        x_train, y_train, x_test, y_test = preprocessor.training_testing_data(processed_data)
        joblib.dump(preprocessor, f"{self.temp_artifact_dir}/preprocessor.pkl")
        self.scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        return x_train, y_train, x_test, y_test, processed_data
    
    def generate_plots(self, model, X_test, y_test):
        # 1. Confusion Matrix
        preds = model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix ({self.current_model_name})")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f"{self.temp_artifact_dir}/confusion_matrix.png")
        plt.close()

        # 2. Feature Importance (Handles Trees AND Logistic Regression)
        plt.figure(figsize=(10,6))
       
        if hasattr(model, 'feature_importances_'):
            # Trees (XGB, LGBM, RF)
            importance = model.feature_importances_
            feat_df = pd.DataFrame({'Feature': X_test.columns, 'Importance': importance})
            feat_df = feat_df.sort_values(by='Importance', ascending=False).head(20)
            sns.barplot(x='Importance', y='Feature', data=feat_df)
           
        elif hasattr(model, 'coef_'):
            # Logistic Regression (Coefficients)
            importance = model.coef_[0]
            feat_df = pd.DataFrame({'Feature': X_test.columns, 'Importance': importance})
            # Sort by absolute value to find biggest drivers
            feat_df['Abs_Importance'] = feat_df['Importance'].abs()
            feat_df = feat_df.sort_values(by='Abs_Importance', ascending=False).head(20)
            sns.barplot(x='Importance', y='Feature', data=feat_df)
           
        plt.title(f"Feature Importance ({self.current_model_name})")
        plt.savefig(f"{self.temp_artifact_dir}/feature_importance.png")
        plt.close()


    def calculate_metrics(self, y_true, y_pred, y_probs):
        return{
            "accuracy": accuracy_score(y_true,y_pred),
            "recall" : recall_score(y_true, y_pred),
            "precision": precision_score(y_true,y_pred),
            "f1": f1_score(y_true,y_pred),
            "roc_auc": roc_auc_score(y_true,y_pred)
        }
    
    def get_model_instance(self, model_name, trial=None, best_params=None, base_scale_pos_weight=1.0):
        """
        Factory to create model instances.
        Now handles robust Logistic Regression tuning and dynamic scale_pos_weight.
        """
        params = best_params if best_params else config[model_name]['params'].copy()
        space = config[model_name]['search_space']

        # Suggest Params if trial exists
        if trial:
            if model_name == 'lightgbm':
                params['num_leaves'] = trial.suggest_int('num_leaves', *space['num_leaves'])
                # Notice we added log=True here manually based on your requirement
                params['learning_rate'] = trial.suggest_float('learning_rate', *space['learning_rate'], log=True)
                params['n_estimators'] = trial.suggest_int('n_estimators', *space['n_estimators'])
                params['lambda_l1'] = trial.suggest_float('lambda_l1', *space['lambda_l1'], log=True)
                params['lambda_l2'] = trial.suggest_float('lambda_l2', *space['lambda_l2'], log=True)
                params['feature_fraction'] = trial.suggest_float('feature_fraction', *space['feature_fraction'])
                params['bagging_fraction'] = trial.suggest_float('bagging_fraction', *space['bagging_fraction'])
                params['bagging_freq'] = trial.suggest_int('bagging_freq', *space['bagging_freq'])
               
            elif model_name == 'xgboost':
                params['n_estimators'] = trial.suggest_int('n_estimators', *space['n_estimators'])
                params['max_depth'] = trial.suggest_int('max_depth', *space['max_depth'])
                params['learning_rate'] = trial.suggest_float('learning_rate', *space['learning_rate'], log=True)
                params['subsample'] = trial.suggest_float('subsample', *space['subsample'])
                params['colsample_bytree'] = trial.suggest_float('colsample_bytree', *space['colsample_bytree'])
                params['gamma'] = trial.suggest_float('gamma', *space['gamma'])
               
                # Dynamic handling of Scale Pos Weight (Multiplier from Config * Calculated Base)
                mult_min, mult_max = space['scale_pos_weight_multipliers']
                multiplier = trial.suggest_float('scale_pos_weight_multiplier', mult_min, mult_max)
                params['scale_pos_weight'] = base_scale_pos_weight * multiplier

            elif model_name == 'random_forest':
                params['n_estimators'] = trial.suggest_int('n_estimators', *space['n_estimators'])
                params['max_depth'] = trial.suggest_int('max_depth', *space['max_depth'])
                params['min_samples_split'] = trial.suggest_int('min_samples_split', *space['min_samples_split'])
                params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', *space['min_samples_leaf'])
                params['max_features'] = trial.suggest_categorical('max_features', space['max_features'])
                params['class_weight'] = trial.suggest_categorical('class_weight', space['class_weight'])
                params['criterion'] = trial.suggest_categorical('criterion', space['criterion'])

            elif model_name == 'logistic_regression':
                # 1. Tune C (Regularization Strength)
                params['C'] = trial.suggest_float('C', *space['C'], log=True)
               
                # 2. Tune Penalty
                penalty_choice = trial.suggest_categorical('penalty', space['penalty'])
                params['penalty'] = penalty_choice
               
                # 3. Handle ElasticNet Specifics
                if penalty_choice == 'elasticnet':
                    # l1_ratio is ONLY used if penalty is elasticnet
                    params['l1_ratio'] = trial.suggest_float('l1_ratio', *space['l1_ratio'])
                else:
                    # Remove l1_ratio if not elasticnet (prevents errors)
                    params.pop('l1_ratio', None)

        # Initialize
        if model_name == 'lightgbm': return lgb.LGBMClassifier(**params)
        if model_name == 'xgboost': return XGBClassifier(**params)
        if model_name == 'random_forest': return RandomForestClassifier(**params)
        if model_name == 'logistic_regression': return LogisticRegression(**params)
       
        raise ValueError(f"Unknown model type: {model_name}")
    
    def objective(self, trial, model_name, X_train, y_train, X_test, y_test, base_scale_pos_weight):
        # 1. Get the Model with suggested params
        model = self.get_model_instance(model_name, trial, base_scale_pos_weight=base_scale_pos_weight)
       
        with mlflow.start_run(nested=True, run_name=f"Trial_{model_name}_{trial.number}"):
           
            # 2. FIT THE MODEL (Handle specifics for Boosting vs others)
            if model_name == 'lightgbm':
                # LightGBM specific: Early Stopping + Eval Set
                callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    eval_metric="auc",
                    callbacks=callbacks
                )
            else:
                # Random Forest & Logistic Regression (Standard Fit)
                model.fit(X_train, y_train)

            # 3. GENERATE PREDICTIONS
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

            # 4. CALCULATE ALL METRICS (Using your helper function)
            metrics = self.calculate_metrics(y_test, preds, probs)

            # Log all parameters (Hyperparameters)
            mlflow.log_params(model.get_params())
            # Log all metrics (Accuracy, F1, AUC, etc.)
            mlflow.log_metrics(metrics)

            return metrics["roc_auc"]

    # def objective(self, trial, X_train, y_train, X_test, y_test, base_scale_pos_weight):
    #     # ========Random forest param =======
    #     # param = { 
    #     #      # 1. Trees: Unlike XGBoost, more trees rarely hurt (just slower) 
    #     #     'n_estimators': trial.suggest_int('n_estimators', 100, 2000), 
    #     #     # 2. Depth: Crucial for overfitting. # If None, it memorizes the training set (bad). Limit it. 
    #     #     'max_depth': trial.suggest_int('max_depth', 5, 35), 
    #     #     # 3. Split Rules: Larger numbers = Simpler/Smoother trees (Less overfitting) 
    #     #     'min_samples_split': trial.suggest_int('min_samples_split', 2, 30), 
    #     #     'min_samples_leaf': trial.suggest_int('min_samples_leaf', 3, 15), 
    #     #     # 4. Features: How many features to look at per split? # 'sqrt' is standard, but tuning it helps. 
    #     #     'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']), 
    #     #     # 5. Imbalance Handling (Built-in!) # 'balanced' automatically weights classes inversely to frequency 
    #     #     'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample']), 
    #     #     'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']), 
    #     #     'random_state': 22, 
    #     #     'n_jobs': -1 # Use all CPU cores 
    #     # }  
        
        
    #     # ======== XG_boost param -======
    #     # param = {
    #     #     'n_estimators': trial.suggest_int('n_estimators', 300, 1800),
    #     #     'max_depth': trial.suggest_int('max_depth', 3, 8),
    #     #     'learning_rate': trial.suggest_float('learning_rate', 0.00001, 0.001),
    #     #     'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    #     #     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    #     #     'gamma': trial.suggest_float('gamma', 0, 5),
    #     #     'scale_pos_weight': trial.suggest_float('scale_pos_weight', base_scale_pos_weight * 1.2, base_scale_pos_weight * 3),
    #     #     'objective': 'binary:logistic',
    #     #     'tree_method': 'hist',
    #     #     'random_state': 22,
    #     #     'n_jobs': -1
    #     # }

    #     # ========== LGB Classifer ============
    #     param = {
    #         'objective': 'binary',
    #         'metric': 'auc',
    #         'verbosity': -1,
    #         'boosting_type': 'gbdt',
           
    #         # 1. Leaf-wise control (Crucial for LGBM)
    #         # num_leaves should be < 2^(max_depth)
    #         'num_leaves': trial.suggest_int('num_leaves', 20, 150),
    #         'max_depth': trial.suggest_int('max_depth', 3, 15),
           
    #         # 2. Regularization (Prevents overfitting on small Telco data)
    #         'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
    #         'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
           
    #         # 3. Sampling
    #         'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
    #         'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
    #         'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
           
    #         # 4. Learning
    #         'learning_rate': trial.suggest_float('learning_rate', 0.0005, 0.02, log=True),
    #         'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
           
    #         # 5. Imbalance
    #         'is_unbalance': True,  # LGBM's built-in handling
    #     }

    #     with mlflow.start_run(nested=True, run_name=f"Trial_LGB_{trial.number}"):
    #         # model = XGBClassifier(**param)
    #         # model = RandomForestClassifier(**param)
    #         model = lgb.LGBMClassifier(**param)
    #         callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
    #         model.fit(
    #             X_train, y_train,
    #             eval_set=[(X_test, y_test)],
    #             eval_metric="auc",
    #         )
    #         # model.fit(X_train, y_train)
    #         importance = model.feature_importances_
    #         feature_imp_df = pd.DataFrame({
    #             "feature_name": X_train.columns,
    #             "feature_imp": importance
    #         })
    #         print("Feature importance =======> ",feature_imp_df)
    #         preds = model.predict(X_test)
    #         probs = model.predict_proba(X_test)[:, 1] # Probability for ROC_AUC
    #         # Calculate ALL metrics
    #         metrics = self.calculate_metrics(y_test, preds, probs)

    #          # Log Params & Metrics to MLflow
    #         mlflow.log_params(param)
    #         mlflow.log_metrics(metrics)

    #         return metrics["roc_auc"]
    
    def run_pipeline(self):
        X_train, y_train, X_test, y_test, full_data = self.load_and_validate_data()
       
        best_overall_score = -1
        best_overall_params = None
        best_overall_model_name = None

        # 1. TOURNAMENT LOOP
        models_to_run = config['models_to_evaluate']
        print(f"--- Starting Tournament: {models_to_run} ---")

        for model_name in models_to_run:
            print(f"\n>>> Optimizing: {model_name.upper()}")
            self.current_model_name = model_name
           
            study = optuna.create_study(direction=config['optuna']['direction'])
            study.optimize(
                lambda trial: self.objective(trial, model_name, X_train, y_train, X_test, y_test,self.scale_pos_weight),
                n_trials=config['optuna']['n_trials']
            )
           
            print(f"Best {model_name} AUC: {study.best_value:.4f}")
           
            if study.best_value > best_overall_score:
                best_overall_score = study.best_value
                best_overall_params = study.best_params
                best_overall_model_name = model_name

        # 2. TRAIN CHAMPION
        print(f"\n=============================================")
        print(f"🏆 WINNER: {best_overall_model_name.upper()} (AUC: {best_overall_score:.4f})")
        print(f"=============================================")
       
        # 2. Train Champion
        print("--- Training Champion Model ---")
        self.current_model_name = best_overall_model_name
        self.train_champion(best_overall_model_name, best_overall_params, X_train, y_train, X_test, y_test, full_data)
    
    def train_champion(self, model_name, params, X_train, y_train, X_test, y_test, full_data):
       
        # Merge static params with optimized params
        final_params = config[model_name]['params'].copy()
        final_params.update(params)

        with mlflow.start_run(run_name=f"Champion_{model_name}") as run:
           
            # A. Train
            model = self.get_model_instance(model_name, best_params=final_params)
            model.fit(X_train, y_train)
           
            # B. Metrics
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]
            metrics = self.calculate_metrics(y_test, preds, probs)
           
            # C. Generate Plots (Saved to self.temp_artifact_dir)
            self.generate_plots(model, X_test, y_test)
           
            # D. LOGGING TO MLFLOW
            mlflow.log_params(final_params)
            mlflow.log_metrics(metrics)
            # train_ds = mlflow.data.from_pandas(full_data, source="processed_telco_data")
            mlflow.log_input(mlflow.data.from_pandas(full_data), context="training")
           
            # --- TAGGING (Requirement 2) ---
            mlflow.set_tag("model_algorithm", model_name)
            mlflow.set_tag("candidate_status", "champion")

            # --- ARTIFACT BUNDLING (Requirement 1) ---
            # This dictionary tells MLflow: "Take these local files and put them INSIDE the model folder"\
            print(self.temp_artifact_dir)
            mlflow.log_artifact(f"{self.temp_artifact_dir}/preprocessor.pkl", artifact_path="data")
           
            # 2. Log Plots
            if os.path.exists(f"{self.temp_artifact_dir}/confusion_matrix.png"):
                mlflow.log_artifact(f"{self.temp_artifact_dir}/confusion_matrix.png", artifact_path="plots")
           
            if os.path.exists(f"{self.temp_artifact_dir}/feature_importance.png"):
                mlflow.log_artifact(f"{self.temp_artifact_dir}/feature_importance.png", artifact_path="plots")

            # Select correct logger based on model type
           # Register correct flavor
            sig = infer_signature(X_test, preds)
           
            if model_name == 'lightgbm':
                mlflow.lightgbm.log_model(model, artifact_path="model", signature=sig, registered_model_name=config['model_name'])
            elif model_name == 'xgboost':
                mlflow.xgboost.log_model(model, artifact_path="model", signature=sig, registered_model_name=config['model_name'])
            elif model_name in ['random_forest', 'logistic_regression']:
                # Both RF and LogReg use standard sklearn logging
                mlflow.sklearn.log_model(model, artifact_path="model", signature=sig, registered_model_name=config['model_name'])
            print(f"Champion Registered: {config['model_name']}")
            print("Cleanup: Removing temp artifacts.")
            shutil.rmtree(self.temp_artifact_dir)

    # def train_and_register(self, params, X_train, y_train, X_test, y_test, full_data):
    #     with mlflow.start_run(run_name="Champion_Model_Candidate_LGB") as run:
            
    #         # model = XGBClassifier(**params) 
    #         # model = RandomForestClassifier(**params)
    #         model = lgb.LGBMClassifier(**params)
    #         model.fit(X_train, y_train)
           
    #         preds = model.predict(X_test)
    #         probs = model.predict_proba(X_test)[:, 1]
           
    #         # Calculate & Log ALL Metrics for the Champion
    #         metrics = self.calculate_metrics(y_test, preds, probs)
    #         mlflow.log_params(params)
    #         mlflow.log_metrics(metrics)
           
    #         # Log Data & Model
    #         train_ds = mlflow.data.from_pandas(full_data, source="processed_telco_data")
    #         mlflow.log_input(train_ds, context="training")
           
    #         signature = infer_signature(X_test, preds)
    #         # model_info = mlflow.sklearn.log_model(
    #         #     sk_model=model,
    #         #     artifact_path="model",
    #         #     signature=signature,
    #         #     registered_model_name=self.model_name
    #         # )

    #         model_info = mlflow.lightgbm.log_model(
    #             lgb_model=model,
    #             artifact_path="model",
    #             signature=signature,
    #             registered_model_name=self.model_name
    #         )
           
    #         # model_info = mlflow.xgboost.log_model(
    #         #     xgb_model=model,
    #         #     artifact_path="model",
    #         #     signature=signature,
    #         #     registered_model_name=self.model_name
    #         # )
    #         print(f"Candidate Model Registered. Metrics: {metrics}")
           
    #         # Pass the ROC_AUC to the promotion logic
    #         self.evaluate_and_promote(
    #             new_version=model_info.registered_model_version,
    #             new_metrics=metrics
    #         )

    # def data_for_testing(self,input_json):
    #     print("Will do later")
    
    # def evaluate_and_promote(self, new_version, new_metrics):
    #     print("--- Evaluating for Promotion ---")
       
    #     # Primary Metric for Decision Making
    #     DECISION_METRIC = "roc_auc"
    #     new_score = new_metrics[DECISION_METRIC]

    #     try:
    #         prod_models = self.client.get_latest_versions(self.model_name, stages=["Production"])
    #     except:
    #         prod_models = []

    #     if not prod_models:
    #         print("No Production model found. Promoting immediately.")
    #         self.promote_to_production(new_version)
    #         return

    #     current_prod = prod_models[0]
    #     prod_run = self.client.get_run(current_prod.run_id)
    #     prod_score = prod_run.data.metrics.get(DECISION_METRIC, 0.0)

    #     print(f"Comparing {DECISION_METRIC}:")
    #     print(f"New Model (v{new_version}): {new_score:.4f}")
    #     print(f"Production (v{current_prod.version}): {prod_score:.4f}")

    #     # Strict Improvement Check
    #     if new_score > prod_score:
    #         print(">>> IMPROVEMENT DETECTED. Promoting.")
    #         self.promote_to_production(new_version)
    #         self.export_model()
    #     else:
    #         print(f">>> NO IMPROVEMENT in {DECISION_METRIC}. Keeping in Staging.")
    #         self.client.transition_model_version_stage(
    #             name=self.model_name,
    #             version=new_version,
    #             stage="Staging"
    #         )

    # def promote_to_production(self, version):
    #     self.client.transition_model_version_stage(
    #         name=self.model_name,
    #         version=version,
    #         stage="Production",
    #         archive_existing_versions=True
    #     )
    #     print(f"Success! Version {version} is now Production.")
    
    

    # def export_model(self):
    #     MODEL_NAME = "TelcoChurnModel-DB-LGB"
    #     STAGE = "Production"
    #     EXPORT_FOLDER = "./my_production_model_lgb"
    #     ZIP_FILENAME = "production_model_pack"
    #     print(f"--- Looking for '{STAGE}' model: {MODEL_NAME} ---")

    #     # 1. Clean up old folders if they exist
    #     if os.path.exists(EXPORT_FOLDER):
    #         shutil.rmtree(EXPORT_FOLDER)
    #     if os.path.exists(f"{ZIP_FILENAME}.zip"):
    #         os.remove(f"{ZIP_FILENAME}.zip")

    #     try:
    #         # 2. THE MAGIC PART
    #         # This one line finds the correct folder inside mlruns and copies it to EXPORT_FOLDER
    #         model_uri = f"models:/{MODEL_NAME}/{STAGE}"
        
    #         print(f"Downloading artifacts from: {model_uri} ...")
    #         mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=EXPORT_FOLDER)
        
    #         print("Success! Model files downloaded.")
        
    #         # 3. Zip it up
    #         shutil.make_archive(ZIP_FILENAME, 'zip', EXPORT_FOLDER)
    #         print(f"\nCREATED ZIP FILE: {os.path.abspath(ZIP_FILENAME + '.zip')}")
    #         print("You can now send this zip file to your Linux machine.")

    #     except Exception as e:
    #         print("\n[ERROR] Could not find the model.")
    #         print("Did you promote a model to Production yet?")
    #         print(f"Error details: {e}")
        
    
  # Call the train_model method

if __name__ == "__main__":
    pipeline = ChurnModelPipeline()
    pipeline.run_pipeline()
    # pipeline.export_model()