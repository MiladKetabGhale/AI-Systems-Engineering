# Housing Price Prediction System Documentation

## Project Overview

This project is a modular, end-to-end pipeline system designed for predicting house prices based on various input features. It automates data ingestion, processing, machine learning model training, evaluation, and results saving. The system is configurable through a YAML file, allowing users to specify data paths, models, evaluation metrics, and more. The project follows best practices with modularity, separation of concerns, error handling, and CI/CD integration (in progress).

---

## System Topology

The system consists of two main components:

1. **Preprocessing Component (ETL)**: Located in `Housing_Data_Processing`, it handles data ingestion, transformation, and storage.
2. **ML Training Component**: Located in `Model_Training`, it trains machine learning models, evaluates their performance, and saves results.

---

## Component Breakdown

### 1. Parsing Component

Located in `Parser`, this component consists of two modules responsible for configuration parsing and error handling.

#### `errors.py`
- Defines custom error classes for robust error handling across the system:
  - **`ConfigError`**: Raised for issues in the configuration file.
  - **`DataValidationError`**: Raised for data validation errors.
  - **`ModelInitializationError`**: Raised for errors in model initialization.
  - **`TrainingError`**: Raised for errors during model training.
  - **`EvaluationError`**: Raised for errors during model evaluation.
  - **`FileHandlingError`**: Raised for errors in file and directory handling.

#### `parser.py`
- **`parse_config`**: Parses and validates the YAML configuration file.
- **`get_model_class`**: Retrieves the model class specified in the config file.
- **`validate_hyperparameters`**: Validates model hyperparameters based on the selected model’s allowed parameters.

---

### 2. ETL Component (Preprocessing)

Located in `Housing_Data_Processing`, this component manages ETL processes, including data ingestion, transformation, and storage. It includes the following modules:

#### `data_ingestion.py`
- **`download_data`**: Downloads the dataset from a URL or uses a local file.
- **`load_data`**: Loads the housing data into a Pandas DataFrame.

#### `data_storage.py`
- **`save_labels`**: Saves training labels to a CSV file.
- **`save_transformed_data`**: Saves transformed training data to a CSV file.
- **`save_test_data`**: Saves testing data and labels to separate CSV files.

#### `base_data_transformation.py`
- **Base Class**: Defines an abstract base class for data transformations, specifying the methods `clean_data` and `transform_features` which are implemented by specific transformation logic classes.

#### `california_housing_transformation.py`
- Extends `BaseDataTransformation` to provide specific transformations for the California Housing dataset.
  - **`clean_data`**: Performs data cleaning and stratified sampling.
  - **`transform_features`**: Transforms features and applies scaling and encoding.

#### `etl_main.py`
- **Main ETL Pipeline Script**: 
  - Parses the configuration file and loads the transformation class.
  - Runs the ingestion, transformation, and storage processes.
  - Dynamically updates file paths for training and testing data.

---

### 3. ML Training Component

Located in `Model_Training`, this component handles model training, evaluation, and result saving. It includes:

#### `trainer.py`
- **`train_model`**: Trains a model using cross-validation and specified hyperparameters.
- **`evaluate_model`**: Evaluates the model based on specified metrics and saves results.
- **`create_results_directory`**: Creates a directory for storing results based on model name and timestamp.
- **`save_run_summary`**: Saves a JSON summary of the model run, including metrics and best parameters.

#### `main.py`
- **Main ML Training Script**:
  - Loads preprocessed data paths and configuration.
  - Initializes, trains, and evaluates the model.
  - Saves evaluation results and best model parameters.

