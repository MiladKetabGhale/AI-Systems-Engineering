# model_saver.py

import os
import joblib

def save_model(best_model, best_params, cv_results, model_name, dir_path="saved_model_params_cvres"):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    model_path = os.path.join(dir_path, f"{model_name}_best_estimator.pkl")
    params_path = os.path.join(dir_path, f"{model_name}_best_params.pkl")
    cv_results_path = os.path.join(dir_path, f"{model_name}_cv_results.pkl")

    joblib.dump(best_model, model_path)
    joblib.dump(best_params, params_path)
    joblib.dump(cv_results, cv_results_path)

    print(f"Model saved as '{model_path}'")
    print(f"Best hyperparameters saved as '{params_path}'")
    print(f"Cross-validation results saved as '{cv_results_path}'")
