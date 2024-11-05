import mlflow
from sklearn.model_selection import GridSearchCV

def train_grid_search_model(model, config, X_train, y_train):
    param_grid = config['base_params']
    
    with mlflow.start_run(nested=True):
        print("Performing grid search for hyperparameter tuning.")
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=config['cv_folds'],
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
