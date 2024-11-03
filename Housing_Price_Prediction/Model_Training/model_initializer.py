# model_initializer.py

from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Mapping of model names to model classes
model_mapping = {
    'lasso': Lasso,
    'ridge': Ridge,
    'linear': LinearRegression,
    'svr': SVR,
    'tree': DecisionTreeRegressor,
    'forest': RandomForestRegressor,
    'sgdr': SGDRegressor
}

# Default hyperparameter grid
hyperparameters_grid = {
    'linear': {'fit_intercept': [True, False]},
    'ridge': {'alpha': [0.01, 0.1, 1, 10, 100], 'solver': ['auto', 'svd', 'cholesky', 'sparse_cg'], 'max_iter': [500, 1000, 2000, 4000]},
    'lasso': {'alpha': [0.01, 0.1, 1, 10, 100, 200, 300, 400], 'max_iter': [2000, 3000], 'tol': [1e-3]},
    'sgdr': {'alpha': [0.01, 0.1, 1, 10], 'max_iter': [500, 1000, 2000], 'tol': [1e-2, 1e-3], 'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'], 'eta0': [0.01, 0.1, 1, 10]},
    'tree': {'max_depth': [3, 4, 5, 6], 'min_samples_split': [5, 10, 15, 20, 25], 'min_samples_leaf': [5, 10, 15, 20], 'max_features': [2, 4, 6, 8], 'criterion': ['squared_error', 'friedman_mse']},
    'forest': {'n_estimators': [3, 6, 9, 12], 'max_depth': [3, 4, 5], 'min_samples_split': [5, 10, 15], 'min_samples_leaf': [5, 10, 15], 'max_features': [2, 4, 6, 8], 'criterion': ['squared_error', 'friedman_mse'], 'bootstrap': [True, False]},
    'svr': {'C': [0.1, 1, 10, 100], 'epsilon': [0.1, 0.5, 1], 'kernel': ['linear'], 'gamma': ['scale', 'auto']}
}

def initialize_model(model_name, model_params=None):
    if model_name not in model_mapping:
        raise ValueError(f"Error: '{model_name}' is not a valid model name.")
    
    print(f"Initializing {model_name.capitalize()} model with parameters: {model_params}")
    model_class = model_mapping[model_name]
    return model_class(**(model_params or {}))
