from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

base_model_classes = {
    'linear': LinearRegression,
    'lasso': Lasso,
    'ridge': Ridge,
    'sgdr': SGDRegressor,
    'tree': DecisionTreeRegressor,
    'svr': SVR
}

def initialize_simple_fit_model(config):
    model_name = config['model_name']
    
    if model_name not in base_model_classes:
        raise ValueError(f"Base model '{model_name}' is not supported for simple fitting.")
    
    base_class = base_model_classes[model_name]
    base_params = {k: v for k, v in config['base_params'].items() if k in base_class().get_params()}
    model = base_class(**base_params)
    
    return model
