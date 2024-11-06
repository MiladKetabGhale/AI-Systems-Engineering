# coding: utf-8

# train_model.py

import mlflow
from sklearn.model_selection import GridSearchCV
import joblib
from datetime import datetime
import os
import json

def train_model(model, config, X_train, y_train):
    """
    Trains the model using the specified configuration and data.

    Args:
        model: The model instance to train.
        config (dict): Configuration dictionary containing model parameters and settings.
        X_train: Training feature data.
        y_train: Training label data.
    
    Returns:
        tuple: Best estimator, best parameters, and cross-validation results.
    """
    param_grid = config['model_params']                  # Use model_params for hyperparameters

    # Ensure single value parameters are wrapped in a list
    for key in param_grid.keys():
        if isinstance(param_grid[key], (int, float)):
            param_grid[key] = [param_grid[key]]          # Wrap in a list if it's a single value

    with mlflow.start_run(nested=True):
        print("Performing GridSearchCV with cross-validation." if config['cv'] > 1 else "Training model without grid search.")

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=config['cv'],
            scoring='neg_mean_squared_error',
            return_train_score=True
        )
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_score", -grid_search.best_score_)
        mlflow.sklearn.log_model(grid_search.best_estimator_, "best_model")

        print("Grid search completed and best model logged to MLflow.")
        return grid_search.best_estimator_, best_params, grid_search.cv_results_

def save_training_artifacts(model, best_params=None, cv_results=None):
    """
    Saves the model, parameters, and cross-validation results (if available) to the specified directory.

    Args:
        model: The trained model to save.
        best_params (dict, optional): The best parameters from training.
        cv_results (dict, optional): Cross-validation results from training.
    
    Returns:
        None
    """
    results_dir = "cvResults_bestModels"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = model.__class__.__name__

    if best_params:
        param_filename = f"{results_dir}/{model_name}_params_{timestamp}.json"
        with open(param_filename, 'w') as f:
            json.dump(best_params, f)
        print(f"Best parameters saved at {param_filename}")

    if cv_results:
        cv_results_filename = f"{results_dir}/{model_name}_cv_results_{timestamp}.pkl"
        joblib.dump(cv_results, cv_results_filename)
        print(f"CV results saved at {cv_results_filename}")

    model_filename = f"{results_dir}/{model_name}_model_{timestamp}.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved at {model_filename}")
