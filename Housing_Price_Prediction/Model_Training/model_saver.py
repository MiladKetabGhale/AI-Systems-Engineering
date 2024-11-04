# model_saver.py

import os
import joblib
import json
from datetime import datetime
import mlflow
import mlflow.sklearn

def save_model(model, params, cv_results, model_name, dir_path="saved_model_params_cvres"):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Generate a timestamp to append to filenames
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model_path = os.path.join(dir_path, f"{model_name}_best_estimator_{timestamp}.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved at '{model_path}'")
    
    # Log the model to MLflow
    mlflow.sklearn.log_model(model, model_name)
    print(f"Model logged to MLflow under the name '{model_name}'")
    
    # Optionally save hyperparameters and CV results as files (can be modified as needed)
    if params:
        params_path = os.path.join(dir_path, f"{model_name}_params_{timestamp}.json")
        with open(params_path, 'w') as f:
            json.dump(params, f)
        mlflow.log_artifact(params_path)
        print(f"Best parameters logged to MLflow as '{params_path}'")
    
    if cv_results:
        cv_results_path = os.path.join(dir_path, f"{model_name}_cv_results__{timestamp}.csv")
        cv_results.to_csv(cv_results_path, index=False)
        mlflow.log_artifact(cv_results_path)
        print(f"CV results logged to MLflow as '{cv_results_path}'")
    

