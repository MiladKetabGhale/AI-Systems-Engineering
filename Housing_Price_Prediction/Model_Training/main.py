# main.py

# Import necessary functions
from parser import parse_config, get_model_class
from training import train_model, evaluate_model, create_results_directory, save_run_summary
import pandas as pd

# Step 1: Parse configuration file
parsed_data = parse_config("config.yaml")

# Step 2: Load data using paths from parsed_data
X_train = pd.read_csv(parsed_data["paths"]["training_data"])
y_train_get = pd.read_csv(parsed_data["paths"]["training_labels"])

y_train = y_train_get.values.ravel()

# Split data into training and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create results directory based on model name, metrics, and timestamp
results_path = create_results_directory(
    model_name=parsed_data["model_name"],
    evaluation_metrics=parsed_data["evaluation_metric"]
)

# Step 4: Initialize the model using the model name from parsed_data
model_class = get_model_class(parsed_data["model_name"])
model = model_class()

# Step 5: Train the model
# We pass individual values from parsed_data into train_model
best_model, best_params, cv_results = train_model(
    model_name=parsed_data["model_name"],   # Pass model name
    model=model,                            # Pass the initialized model
    X_train=X_train,                        # Training features
    y_train=y_train,                        # Training labels
    param_grid=parsed_data["param_grid"],   # Pass hyperparameter grid
    cv=parsed_data["cv"],                   # Cross-validation setting
    evaluation_metrics=parsed_data["evaluation_metric"]
)

# Step 6: Evaluate the model
# Pass individual values from parsed_data into evaluate_model
metrics = evaluate_model(
    model=best_model,                                     # The trained model from train_model
    X_test=X_train,                                       # Test features
    y_test=y_train,                                       # Test labels
    results_path=parsed_data["paths"]["results"],         # Results path from parsed_data
    model_name=parsed_data["model_name"],                 # Model name
    evaluation_metrics=parsed_data["evaluation_metric"],  # parsed evaluation metrics to use
    best_params=best_params,                              # Best parameters from GridSearchCV
    cv_results=cv_results                                 # Cross-validation results from GridSearchCV
)

# Step 7: Save run summary in the results directory
save_run_summary(
    results_path=results_path,
    model_name=parsed_data["model_name"],
    evaluation_metrics=parsed_data["evaluation_metric"],
    best_params=best_params,
    metrics=metrics
)

# Print the evaluation metrics
print("Training and evaluation completed. Metrics:", metrics)
