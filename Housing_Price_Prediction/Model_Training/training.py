# training.py

from sklearn.model_selection import GridSearchCV
import pandas as pd
import mlflow

def train_model(model, X_train, y_train, param_grid=None, cv=5):
    if cv == 1:
        print("Training model without cross-validation.")
        model.fit(X_train, y_train)
        
        # Log training metrics if available
        train_score = model.score(X_train, y_train)
        mlflow.log_metric("train_score", train_score)
        
        return model, {}, {}
    
    elif param_grid:
        print("Performing grid search for hyperparameter tuning.")
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error', return_train_score=True)
        grid_search.fit(X_train, y_train)
        
        # Log only the final best parameters after training
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_score", -grid_search.best_score_)
        mlflow.sklearn.log_model(grid_search.best_estimator_, "best_model")
        
        # Save cv_results_ as a CSV file
        cv_results_df = pd.DataFrame(grid_search.cv_results_)
        cv_results_path = "cv_results.csv"
        cv_results_df.to_csv(cv_results_path, index=False)
        
        mlflow.log_artifact("cv_results.csv")
        print("Grid search completed.")
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_
    else:
        print("Training model without hyperparameter tuning.")
        model.fit(X_train, y_train)
        return model, {}, {}
