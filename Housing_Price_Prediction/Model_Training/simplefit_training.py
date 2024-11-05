import mlflow

def train_simple_fit_model(model, config, X_train, y_train):
    with mlflow.start_run(nested=True):
        print("Training model without hyperparameter tuning.")
        model.fit(X_train, y_train)

        mlflow.log_params(config['base_params'])
        train_score = model.score(X_train, y_train)
        mlflow.log_metric("train_score", train_score)

        mlflow.sklearn.log_model(model, "trained_model")
        print("Model training completed and logged to MLflow.")
    
    return model
