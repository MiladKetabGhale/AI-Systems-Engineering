import mlflow

def train_ensemble_model(model, config, X_train, y_train):
    with mlflow.start_run(nested=True):
        print("Training ensemble model without hyperparameter tuning.")
        model.fit(X_train, y_train)

        mlflow.log_params(config['ensemble_params'])
        mlflow.log_param("ensemble_name", config['ensemble_name'])
        mlflow.log_param("ensemble_base_model", config['ensemble_base_model'])

        train_score = model.score(X_train, y_train)
        mlflow.log_metric("train_score", train_score)

        mlflow.sklearn.log_model(model, "ensemble_model")
        print("Ensemble model training completed and logged to MLflow.")
    
    return model
