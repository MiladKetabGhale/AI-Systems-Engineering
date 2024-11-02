# etl_main.py

import os
import sys
from data_ingestion import DataIngestion
from data_storage import DataStorage

def load_config(config_file):
    """Loads configuration parameters from a file."""
    print("Attempting to load config file: {}".format(config_file))  # Debugging line
    config = {}
    if not os.path.exists(config_file):
        raise FileNotFoundError("Configuration file '{}' does not exist.".format(config_file))

    with open(config_file, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, value = line.split('=', 1)
            config[key.strip()] = value.strip()

    return config

def run_etl_pipeline(config, TransformationClass):
    """Runs the ETL pipeline with parameters from the config dictionary."""
    # URL is mandatory; raise an error if it's not provided
    data_url = config.get("data_url")
    if not data_url:
        raise ValueError("The 'data_url' must be specified in the config file.")

    # Use default data_path if not specified
    data_path = config.get("data_path", "default_data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    # Allow optional user-defined save_path or default to `./cleanDatasets`
    save_path = config.get("save_path", "./cleanDatasets")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Ingest data
    ingestion = DataIngestion(data_url, data_path)
    ingestion.download_data()
    housing = ingestion.load_data()

    # Use the TransformationClass for this specific dataset
    transformation = TransformationClass(save_path=save_path)
    housing, housing_labels = transformation.clean_data(housing)
    housing_prepared = transformation.transform_features(housing)

    # Store data
    storage = DataStorage(save_path=save_path)
    storage.save_labels(housing_labels)
    storage.save_transformed_data(housing_prepared)

if __name__ == "__main__":
    # Check for command-line argument for config file
    if len(sys.argv) != 2:
        print("Usage: python etl_main.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    config = load_config(config_file)

    # Import transformation class dynamically
    transformation_module = os.getenv('TRANSFORMATION_MODULE', 'california_housing_transformation')
    transformation_class = os.getenv('TRANSFORMATION_CLASS', 'CaliforniaHousingTransformation')

    # Import the specified transformation class dynamically
    module = __import__(transformation_module)
    TransformationClass = getattr(module, transformation_class)
    
    run_etl_pipeline(config, TransformationClass)

