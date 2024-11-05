# initialize_model.py

from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# Mapping of model names to classes
MODEL_CLASSES = {
    'bagging': BaggingRegressor,
    'adaboost': AdaBoostRegressor,
    'gradient_boosting': GradientBoostingRegressor,
    'xgboost': XGBRegressor,
    'lightgb': LGBMRegressor,
    'linear': LinearRegression,
    'lasso': Lasso,
    'ridge': Ridge,
    'sgdr': SGDRegressor,
    'tree': DecisionTreeRegressor,
    'svr': SVR
}

def initialize_model(config):
    """
    Initializes the model based on the configuration and prepares it for training with or without GridSearchCV.
    """
    model_name = config['model_name']
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unsupported model type: {model_name}")

    model_class = MODEL_CLASSES[model_name]
    model_params = {k: v for k, v in config['model_params'].items() if k in model_class().get_params()}
    
    # Initialize model
    model = model_class(**model_params)

    # Prepare the parameter grid for GridSearchCV if cv > 1
    param_grid = config['model_params'] if config['cv'] > 1 else None
    return model, param_grid
