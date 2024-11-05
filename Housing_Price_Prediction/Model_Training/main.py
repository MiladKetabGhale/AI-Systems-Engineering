import argparse
import mlflow
import pandas as pd
from config_loader import load_config
from ensemble_initialization import initialize_ensemble_model
from gridsearch_initialization import initialize_grid_search_model
from simplefit_initialization import initialize_simple_fit_model
from ensemble_training import train_ensemble_model
from gridsearch_training import train_grid_search_model
from simplefit_training import train_simple_fit_model
from evaluation import evaluate_model

def main(config_path):
    mlflow.set_experiment("Housing Price Prediction")
    
    with mlflow.start_run(run_name="Full Pipeline") as parent_run:
        config = load_config(config_path)
        mlflow.log_params(config)

        # Load training and test data
        X_train = pd.read_csv(config['training_data_path'])
        y_train = pd.read_csv(config['labels_data_path']).values.ravel()
        X_test = X_train
        y_test = y_train

        # Initialize and train the model based on training_type
        trained_model = None
        if config['training_type'] == "ensemble":
            with mlflow.start_run(run_name="Ensemble Initialization and Training", nested=True):
                model = initialize_ensemble_model(config)
                trained_model = train_ensemble_model(model, config, X_train, y_train)

        elif config['training_type'] == "gridsearch":
            with mlflow.start_run(run_name="GridSearch Initialization and Training", nested=True):
                model = initialize_grid_search_model(config)
                trained_model, _, _ = train_grid_search_model(model, config, X_train, y_train)

        elif config['training_type'] == "simplefit":
            with mlflow.start_run(run_name="SimpleFit Initialization and Training", nested=True):
                model = initialize_simple_fit_model(config)
                trained_model = train_simple_fit_model(model, config, X_train, y_train)

        # Model Evaluation
        if trained_model:
            with mlflow.start_run(run_name="Model Evaluation", nested=True):
                predictions, evaluation_metrics = evaluate_model(trained_model, config, X_test, y_test)
                mlflow.log_metrics(evaluation_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ML pipeline with a specified config file.")
    parser.add_argument("-c", "--config", required=True, help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config)
