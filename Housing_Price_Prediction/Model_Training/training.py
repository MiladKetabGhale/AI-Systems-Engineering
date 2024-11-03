# training.py

from sklearn.model_selection import GridSearchCV

def train_model(model, X_train, y_train, param_grid=None, cv=5):
    if cv == 1:
        print("Training model without cross-validation.")
        model.fit(X_train, y_train)
        return model, {}, {}
    elif param_grid:
        print("Performing grid search for hyperparameter tuning.")
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error', return_train_score=True)
        grid_search.fit(X_train, y_train)
        print("Grid search completed.")
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_
    else:
        print("Training model without hyperparameter tuning.")
        model.fit(X_train, y_train)
        return model, {}, {}
