# trainer.py
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from joblib import dump
import json
import os
from datetime import datetime

# Define a mapping of available scoring functions for regression
SCORING_FUNCTIONS = {
    "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
    "MSE": make_scorer(mean_squared_error, greater_is_better=False),
    "RMSE": make_scorer(lambda y, y_pred: mean_squared_error(y, y_pred, squared=False), greater_is_better=False),
    "R2": make_scorer(r2_score),
    "Explained Variance": make_scorer(explained_variance_score)
}

def train_model(model_name, model, X_train, y_train, param_grid, cv, evaluation_metrics):
    """
    Trains the model based on cross-validation setting.
    Returns: best_model, best_params, cv_results
    """
    
    print("Starting training for {}... This may take a while.".format(model_name))
    # Convert evaluation metrics to scoring dictionary for GridSearchCV
    scoring = {metric: SCORING_FUNCTIONS[metric] for metric in evaluation_metrics if metric in SCORING_FUNCTIONS}

    if not scoring:
        raise ValueError("No valid scoring metrics found in the evaluation metrics list.")

    # Choose the first metric in evaluation_metrics as the primary metric for optimization
    refit_metric = evaluation_metrics[0]
    
    if cv > 1:
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_results = grid_search.cv_results_
    else:
        best_model = model.fit(X_train, y_train)
        best_params = model.get_params()
        cv_results = None
    print(f"Training for {model_name} completed successfully.")
    return best_model, best_params, cv_results


def evaluate_model(model, X_test, y_test, results_path, model_name, evaluation_metrics, best_params=None, cv_results=None):
    """
    Evaluates the model based on specified metrics and saves results to the specified path.
    
    Args:
    - model: Trained model.
    - X_test, y_test: Test data and labels.
    - results_path (str): Path to save evaluation results and model.
    - model_name (str): Name of the model.
    - evaluation_metric (list): List of metrics to calculate (from configuration).
    - best_params (dict): Best hyperparameters if GridSearchCV was used, else None.
    - cv_results (dict): Cross-validation results if GridSearchCV was used, else None.
    
    Returns:
    - metrics (dict): Calculated metrics based on the specified list.
    """
    os.makedirs(results_path, exist_ok=True)
    y_pred = model.predict(X_test)
    
    # Calculate specified evaluation metrics
    metrics = {}
    for metric in evaluation_metrics:
        if metric == "MAE":
            metrics["MAE"] = mean_absolute_error(y_test, y_pred)
        elif metric == "MSE":
            metrics["MSE"] = mean_squared_error(y_test, y_pred)
        elif metric == "RMSE":
            metrics["RMSE"] = mean_squared_error(y_test, y_pred, squared=False)
        elif metric == "R2":
            metrics["R2"] = r2_score(y_test, y_pred)
        elif metric == "Explained Variance":
            metrics["Explained Variance"] = explained_variance_score(y_test, y_pred)
        else:
            raise ValueError(f"Unsupported evaluation metric '{metric}' specified.")

    # Save metrics, best parameters, and cross-validation results
    dump(metrics, os.path.join(results_path, f"{model_name}_metrics.joblib"))

    # Save cv_results with joblib
    if cv_results:
        dump(cv_results, os.path.join(results_path, f"{model_name}_cv_results.joblib"))

    # Save the best parameters with joblib
    if best_params:
        dump(best_params, os.path.join(results_path, f"{model_name}_best_params.joblib"))

    # Save the trained model
    dump(model, os.path.join(results_path, f"{model_name}_best_model.joblib"))
    print(f"Evaluation for {model_name} completed successfully.")
    return metrics


def create_results_directory(model_name, evaluation_metrics):
    """
    Creates a results directory based on model name, evaluation metrics, and timestamp.
    Returns the path to the created directory.
    """
    # Format the evaluation metrics and timestamp for the directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_str = "_".join(evaluation_metrics)
    results_path = os.path.join("results", model_name, metrics_str, timestamp)
    os.makedirs(results_path, exist_ok=True)
    return results_path

def save_run_summary(results_path, model_name, evaluation_metrics, best_params, metrics):
    """
    Saves a summary of the run in JSON format in the results directory.
    """
    summary = {
        "model_name": model_name,
        "evaluation_metrics": evaluation_metrics,
        "best_params": best_params,
        "metrics": metrics,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    # Save the summary to a JSON file in the results directory
    with open(os.path.join(results_path, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
