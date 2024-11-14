# etl_main.py

import os
import sys
from data_ingestion import DataIngestion
from data_storage import DataStorage
from data_ingestion import DataIngestion
from data_storage import DataStorage
from Parser.parser import parse_config  # Assuming parser is in the `Parser` directory
from Parser.errors import ConfigError, DataValidationError
import importlib.util

def run_etl_pipeline(config, TransformationClass, dataset_name):
    """Runs the ETL pipeline with parameters from the config dictionary."""
    # Retrieve the data ingestion source from the parsed config
    data_ingestion_source = config["paths"]["data_ingestion_source"]
    if not data_ingestion_source:
        raise ValueError("The 'data_ingestion_source' must be specified in the config file.")

    # Set up data path for downloaded or ingested data
    data_path = config["paths"].get("training_data") or "./cleanDatasets"
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Set save paths for processed data and labels
    save_data_path = config["paths"].get("save_data_path", "").strip() or "./cleanDatasets"
    save_labels_path = config["paths"].get("save_labels_path", "").strip() or save_data_path
    os.makedirs(save_data_path, exist_ok=True)    
    
    # Ingest data
    ingestion = DataIngestion(data_ingestion_source, data_path)
    ingestion.download_data()
    housing = ingestion.load_data()

    # Use the TransformationClass for this specific dataset
    transformation = TransformationClass(save_path=save_data_path)
    housing, housing_labels, housing_test, housing_labels_test = transformation.clean_data(housing)
    housing_prepared = transformation.transform_features(housing)
    housing_test = transformation.transform_features(housing_test)
    
    # Store processed data and labels with dataset_name for dynamic naming
    storage = DataStorage(save_path=save_data_path, dataset_name=dataset_name)
    storage.save_labels(housing_labels)
    storage.save_transformed_data(housing_prepared)
    
    # save the test data
    storage.save_test_data(housing_test, housing_labels_test)
    
    # Check if training data and labels are saved correctly
    training_data_path = os.path.join(save_data_path, f"{storage.dataset_name}_prepared.csv")
    training_labels_path = os.path.join(save_labels_path, f"{storage.dataset_name}_labels.csv")
    
    if not (os.path.isfile(training_data_path) and os.path.isfile(training_labels_path)):
        raise FileNotFoundError(
            f"Training data or labels not found. Expected files at:\n"
            f"{training_data_path}\n{training_labels_path}\n"
            "Please verify preprocessing steps or paths."
        )

    # These files will be checked before evaluation.
    testing_data_path = config["paths"].get("testing_data", "")
    testing_labels_path = config["paths"].get("testing_labels", "")
    if not testing_data_path:
        testing_data_path = os.path.join(save_data_path, f"{storage.dataset_name}_test.csv")
    if not testing_labels_path:
        testing_labels_path = os.path.join(save_labels_path, f"{storage.dataset_name}_test_labels.csv")

    # Update config with test paths for consistency
    config["paths"]["testing_data"] = testing_data_path
    config["paths"]["testing_labels"] = testing_labels_path

def to_pascal_case(snake_str):
    """Converts a snake_case string to PascalCase."""
    components = snake_str.split('_')
    return ''.join(x.capitalize() for x in components)
    
if __name__ == "__main__":
    # Check for command-line argument for config file
    if len(sys.argv) != 2:
        print("Usage: python etl_main.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    config = parse_config(config_file)

    # Retrieve transformation logic path from the parsed config
    transformation_logic_file = config["paths"]["transformation_logic_path"]
    if not transformation_logic_file:
        raise ValueError("The 'transformation_logic_path' must be specified in the config file.")
    
    # Extract dataset name from the transformation logic file for dynamic file naming
    dataset_name = os.path.splitext(os.path.basename(transformation_logic_file))[0]

    # Load the transformation module dynamically from the provided file path
    spec = importlib.util.spec_from_file_location(dataset_name, transformation_logic_file)
    transformation_module = importlib.util.module_from_spec(spec)
    sys.modules[dataset_name] = transformation_module
    spec.loader.exec_module(transformation_module)

    # Infer class name from the module name (assumes PascalCase convention)
    transformation_class_name = to_pascal_case(dataset_name)

    # Attempt to dynamically retrieve the transformation class
    try:
        TransformationClass = getattr(transformation_module, transformation_class_name)
    except AttributeError:
        raise ImportError(f"Failed to find class '{transformation_class_name}' in module '{dataset_name}'.")

    # Run the ETL pipeline
    run_etl_pipeline(config, TransformationClass, dataset_name)

