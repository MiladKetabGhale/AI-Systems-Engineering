# main.py

# Import necessary functions
import os
import sys
from Parser.parser import parse_config, get_model_class
from training import train_model, evaluate_model, create_results_directory, save_run_summary
import pandas as pd

# Step 1: Parse configuration file (from command line argument)
if len(sys.argv) != 2:
    print("Usage: python main.py <config_file>")
    sys.exit(1)

config_file = sys.argv[1]  # Takes config file path from command line
parsed_data = parse_config(config_file)

# Extract the transformation logic path and derive dataset name
transformation_logic_path = parsed_data["paths"].get("transformation_logic_path")
dataset_name = os.path.splitext(os.path.basename(transformation_logic_path))[0]  # Extracts name for naming convention

# Set default directory for preprocessed data
data_dir = "./cleanDatasets"

# Define paths for training data, training labels, testing data, and testing labels
training_data_path = parsed_data["paths"].get("training_data")
if not training_data_path:
    training_data_path = os.path.join(data_dir, f"{dataset_name}_prepared.csv")
    if not os.path.exists(training_data_path):
        raise FileNotFoundError(f"Training data file '{training_data_path}' does not exist.")

training_labels_path = parsed_data["paths"].get("training_labels")
if not training_labels_path:
    training_labels_path = os.path.join(data_dir, f"{dataset_name}_labels.csv")
    if not os.path.exists(training_labels_path):
        raise FileNotFoundError(f"Training labels file '{training_labels_path}' does not exist.")

testing_data_path = parsed_data["paths"].get("testing_data")
if not testing_data_path:
    testing_data_path = os.path.join(data_dir, f"{dataset_name}_test.csv")
    if not os.path.exists(testing_data_path):
        raise FileNotFoundError(f"Testing data file '{testing_data_path}' does not exist.")

testing_labels_path = parsed_data["paths"].get("testing_labels")
if not testing_labels_path:
    testing_labels_path = os.path.join(data_dir, f"{dataset_name}_test_labels.csv")
    if not os.path.exists(testing_labels_path):
        raise FileNotFoundError(f"Testing labels file '{testing_labels_path}' does not exist.")

# Load training data
X_train = pd.read_csv(training_data_path)
y_train = pd.read_csv(training_labels_path).values.ravel()

# Load testing data
X_test = pd.read_csv(testing_data_path)
y_test = pd.read_csv(testing_labels_path).values.ravel()

# Step 3: Create results directory based on model name, metrics, and timestamp
results_path = create_results_directory(
    model_name=parsed_data["model_name"],
    evaluation_metrics=parsed_data["evaluation_metric"]
)

# Step 4: Initialize the model using the model name from parsed_data
model_class = get_model_class(parsed_data["model_name"])
model = model_class()

# Step 5: Train the model
best_model, best_params, cv_results = train_model(
    model_name=parsed_data["model_name"],
    model=model,
    X_train=X_train,
    y_train=y_train,
    param_grid=parsed_data["param_grid"],
    cv=parsed_data["cv"],
    evaluation_metrics=parsed_data["evaluation_metric"]
)

# Step 6: Evaluate the model
metrics = evaluate_model(
    model=best_model,
    X_test=X_test,
    y_test=y_test,
    results_path=parsed_data["paths"]["results"],
    model_name=parsed_data["model_name"],
    evaluation_metrics=parsed_data["evaluation_metric"],
    best_params=best_params,
    cv_results=cv_results
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

