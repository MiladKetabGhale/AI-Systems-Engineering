from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

base_model_classes = {
    'linear': LinearRegression,
    'lasso': Lasso,
    'ridge': Ridge,
    'sgdr': SGDRegressor,
    'tree': DecisionTreeRegressor,
    'svr': SVR
}

def initialize_grid_search_model(config):
    model_name = config['model_name']
    
    if model_name not in base_model_classes:
        raise ValueError(f"Base model '{model_name}' is not supported for grid search.")
    
    base_class = base_model_classes[model_name]
    base_params = {k: v for k, v in config['base_params'].items() if k in base_class().get_params()}
    base_model = base_class(**base_params)
    
    param_grid = config['base_params'] if 'base_params' in config else {}
    model = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=config['cv_folds'],
        scoring='neg_mean_squared_error',
        return_train_score=True
    )
    
    return model
