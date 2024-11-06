import argparse
import mlflow
import pandas as pd
from config_loader import load_config
from initialize_model import initialize_model
from train_model import train_model
from evaluation import evaluate_model

def main(config_path):
    """
    Main entry point for the machine learning pipeline.

    Args:
        config_path (str): The path to the configuration file.
    
    Returns:
        None
    """
    mlflow.set_experiment("Housing Price Prediction")
    
    with mlflow.start_run(run_name="Full Pipeline") as parent_run:
        config = load_config(config_path)
        mlflow.log_params(config)

        # Load training and test data
        X_train = pd.read_csv(config['training_data_path'])
        y_train = pd.read_csv(config['labels_data_path']).values.ravel()
        X_test = X_train  # Use train data as test data if no separate test data is provided
        y_test = y_train

        # Initialize model and parameter grid
        model, param_grid = initialize_model(config)

        # Train model with or without GridSearchCV based on cv value
        with mlflow.start_run(run_name="Training and Evaluation", nested=True):
            trained_model, best_params, cv_results = train_model(model, config, X_train, y_train)

        # Evaluate model
        with mlflow.start_run(run_name="Model Evaluation", nested=True):
            predictions, evaluation_metrics = evaluate_model(trained_model, config, X_test, y_test)
            mlflow.log_metrics(evaluation_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ML pipeline with a specified config file.")
    parser.add_argument("-c", "--config", required=True, help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config)
