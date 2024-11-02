import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class HousingPricePredictor:
    def __init__(self):
        # Mapping of model names to model classes
        self.model_mapping = {
            'lasso': Lasso,
            'ridge': Ridge,
            'linear': LinearRegression,
            'svr': SVR,
            'tree': DecisionTreeRegressor,
            'forest': RandomForestRegressor,
            'sgdr': SGDRegressor
        }

        # Default hyperparameter grid
        self.hyperparameters_grid = {
            'linear': {'fit_intercept': [True, False]},
            'ridge': {'alpha': [0.01, 0.1, 1, 10, 100], 'solver': ['auto', 'svd', 'cholesky', 'sparse_cg'], 'max_iter': [500, 1000, 2000, 4000]},
            'lasso': {'alpha': [0.01, 0.1, 1, 10, 100, 200, 300, 400], 'max_iter': [2000, 3000], 'tol': [1e-3]},
            'sgdr': {'alpha': [0.01, 0.1, 1, 10], 'max_iter': [500, 1000, 2000], 'tol': [1e-2, 1e-3], 'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'], 'eta0': [0.01, 0.1, 1, 10]},
            'tree': {'max_depth': [3, 4, 5, 6], 'min_samples_split': [5, 10, 15, 20, 25], 'min_samples_leaf': [5, 10, 15, 20], 'max_features': [2, 4, 6, 8], 'criterion': ['squared_error', 'friedman_mse']},
            'forest': {'n_estimators': [3, 6, 9, 12], 'max_depth': [3, 4, 5], 'min_samples_split': [5, 10, 15], 'min_samples_leaf': [5, 10, 15], 'max_features': [2, 4, 6, 8], 'criterion': ['squared_error', 'friedman_mse'], 'bootstrap': [True, False]},
            'svr': {'C': [0.1, 1, 10, 100], 'epsilon': [0.1, 0.5, 1], 'kernel': ['linear'], 'gamma': ['scale', 'auto']}
        }
        self.fitted_model = None  # To store the trained model

    def load_config(self, config_file_path):
        config = {
            'training_data_path': None,
            'labels_data_path': None,
            'cv_folds': 5,
            'model_name': None,
            'hyperparameters': {}
        }

        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Configuration file '{config_file_path}' does not exist.")

        with open(config_file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                key, value = line.split('=', 1)
                key, value = key.strip(), value.strip()
                if key == "training_data_path":
                    config['training_data_path'] = value
                elif key == "labels_data_path":
                    config['labels_data_path'] = value
                elif key == "cv_folds":
                    config['cv_folds'] = int(value)
                elif key == "model_name":
                    config['model_name'] = value.lower()
                elif key.startswith("hyperparameter_"):
                    param_name = key[len("hyperparameter_"):].strip()
                    # Detect the appropriate type and wrap in list
                    if value.isdigit():
                        config['hyperparameters'][param_name] = [int(value)]
                    else:
                        try:
                            config['hyperparameters'][param_name] = [float(value)]
                        except ValueError:
                            config['hyperparameters'][param_name] = [value]  # Leave as string in list if not a number

        # Apply defaults if necessary
        if config['model_name'] in self.hyperparameters_grid:
            for param, default_values in self.hyperparameters_grid[config['model_name']].items():
                # Only add the default if it doesn't already exist in user-provided hyperparameters
                config['hyperparameters'].setdefault(param, default_values if isinstance(default_values, list) else [default_values])

        return config
    

    def choose_and_fit(self, model_name, train_set, train_labels, hyperparameters, cv=5):
        if model_name not in self.model_mapping:
            raise ValueError(f"Error: '{model_name}' is not a valid model name.")
        
        model_class = self.model_mapping[model_name]
        model = model_class()

        if cv == 1:
            model.fit(train_set, train_labels)
            return model, {}, {}
        else:
            grid_search = GridSearchCV(model, hyperparameters, cv=cv, scoring='neg_mean_squared_error', return_train_score=True)
            grid_search.fit(train_set, train_labels)
            self.fitted_model = grid_search.best_estimator_  # Store the best model in the class
            return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_

    def fit_and_save(self, model_name, train_set, train_labels, hyperparameters, cv=5, dir_path="saved_model_params_cvres"):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        model_path = os.path.join(dir_path, f"{model_name}_best_estimator.pkl")
        params_path = os.path.join(dir_path, f"{model_name}_best_params.pkl")
        cv_results_path = os.path.join(dir_path, f"{model_name}_cv_results.pkl")

        best_model, best_params, cv_results = self.choose_and_fit(model_name, train_set, train_labels, hyperparameters, cv=cv)

        joblib.dump(best_model, model_path)
        joblib.dump(best_params, params_path)
        joblib.dump(cv_results, cv_results_path)

        print(f"Model saved as '{model_path}'")
        print(f"Best hyperparameters saved as '{params_path}'")
        print(f"Cross-validation results saved as '{cv_results_path}'")

        return best_model, best_params, cv_results

    def load_and_predict(self, model_path, input_data, true_labels=None, compute_errors=True):
        model_name = os.path.basename(model_path).split('_')[0].lower()
        model = joblib.load(model_path)
        predictions = model.predict(input_data)

        evaluation_metrics = {}
        if compute_errors and true_labels is not None:
            evaluation_metrics[f'{model_name}_mean_squared_error'] = mean_squared_error(true_labels, predictions)
            evaluation_metrics[f'{model_name}_root_mean_squared_error'] = np.sqrt(mean_squared_error(true_labels, predictions))
            evaluation_metrics[f'{model_name}_mean_absolute_error'] = mean_absolute_error(true_labels, predictions)
            evaluation_metrics[f'{model_name}_r2_score'] = r2_score(true_labels, predictions)

        return predictions, evaluation_metrics if evaluation_metrics else None


    def predict(self, input_data, true_labels=None, compute_errors=True):
        """
        Generates predictions using the fitted model saved in the class
        """
        if self.fitted_model is None:
            raise ValueError("No model has been fitted yet. Please fit a model first using choose_and_fit.")

        predictions = self.fitted_model.predict(input_data)
        evaluation_metrics = {}

        # If true labels are provided, calculate error metrics
        if compute_errors and true_labels is not None:
            evaluation_metrics['mean_squared_error'] = mean_squared_error(true_labels, predictions)
            evaluation_metrics['root_mean_squared_error'] = np.sqrt(mean_squared_error(true_labels, predictions))
            evaluation_metrics['mean_absolute_error'] = mean_absolute_error(true_labels, predictions)
            evaluation_metrics['r2_score'] = r2_score(true_labels, predictions)

        return predictions, evaluation_metrics if evaluation_metrics else None

