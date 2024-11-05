from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# Ensemble and base model mappings
ensemble_classes = {
    'bagging': BaggingRegressor,
    'adaboost': AdaBoostRegressor,
    'gradient_boosting': GradientBoostingRegressor,
    'xgboost': XGBRegressor,
    'lightgb': LGBMRegressor
}

base_model_classes = {
    'linear': LinearRegression,
    'lasso': Lasso,
    'ridge': Ridge,
    'sgdr': SGDRegressor,
    'tree': DecisionTreeRegressor,
    'svr': SVR
}

def initialize_ensemble_model(config):
    """
    Initializes an ensemble model with the given base model and hyperparameters.
    Applies GridSearchCV if cv > 1.
    """
    ensemble_name = config['ensemble_name']
    base_model_name = config['ensemble_base_model']
    cv = config['cv']
    ensemble_params = config['ensemble_params']
    base_params = config['base_params']

    if ensemble_name not in ensemble_classes:
        raise ValueError(f"Ensemble type '{ensemble_name}' is not supported.")
    if base_model_name not in base_model_classes:
        raise ValueError(f"Base model '{base_model_name}' for ensemble is not supported.")
    
    # Initialize base model with validated base_params
    base_class = base_model_classes[base_model_name]
    filtered_base_params = {k: v for k, v in base_params.items() if k in base_class().get_params()}
    base_model = base_class(**filtered_base_params)
    
    # Initialize ensemble with validated ensemble_params
    ensemble_class = ensemble_classes[ensemble_name]
    filtered_ensemble_params = {k: v for k, v in ensemble_params.items() if k in ensemble_class().get_params()}

    # Convert single values to lists if using GridSearchCV (cv > 1)
    if cv > 1:
        for key, value in filtered_ensemble_params.items():
            if not isinstance(value, list):
                filtered_ensemble_params[key] = [value]  # Wrap single values in a list

        model = GridSearchCV(
            estimator=ensemble_class(base_estimator=base_model),
            param_grid=filtered_ensemble_params,
            cv=cv,
            scoring='neg_mean_squared_error',
            return_train_score=True
        )
    else:
        # For non-GridSearchCV (cv=1), convert any list values to single values
        for key, value in filtered_ensemble_params.items():
            if isinstance(value, list):
                filtered_ensemble_params[key] = value[0]  # Use the first value in the list

        model = ensemble_class(base_estimator=base_model, **filtered_ensemble_params)

    return model
