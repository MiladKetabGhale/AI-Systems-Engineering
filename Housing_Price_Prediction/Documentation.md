# Housing Price Prediction System Documentation

## Project Overview

This project is a modular, end-to-end pipeline system designed for predicting house prices based on various input features. It automates data ingestion, processing, machine learning model training, evaluation, and results saving. The system is configurable through a YAML file, allowing users to specify data paths, models, evaluation metrics, and more. The project follows best practices with modularity, separation of concerns, error handling, and CI/CD integration (in progress).

---

## System Topology

The system consists of five main components:

1. **Configuration Component**: Enables customization of the system through a YAML configuration file. Users can specify data paths, transformation logic, models, hyperparameters, evaluation metrics, and more. This component ensures flexibility and allows the system to adapt to various datasets and use cases without code modifications.
2. **Parsing Component**: Located in Parser, it parses and validates the configuration file. It also includes robust error handling for missing or invalid configurations. This component ensures the integrity of inputs and prepares the system for execution.
3. **Preprocessing Component (ETL)**: Located in `Housing_Data_Processing`, it handles data ingestion, transformation, and storage.
4. **Transformation Logic**: The user must provide their specific business transformation logic module. The system automatically then finds and builds based on this specification by instantiating the generic transformation logic.
4. **ML Training Component**: Located in `Model_Training`, it trains machine learning models, evaluates their performance, and saves results.
---

## Component Breakdown

### 1. Configuration File

The configuration file is a YAML file that allows users to define the behavior of the system. It consists of the following sections:

#### **Paths Section**
This section defines the locations for data ingestion, processing, and saving. Some paths are mandatory, while others are optional.

- **`data_ingestion_source`** (Mandatory): Specifies the source of the dataset (URL or local path).
- **`transformation_logic_path`** (Mandatory): Path to the transformation logic file, which defines dataset-specific preprocessing rules.
- **`training_data`** (Optional): Path to preprocessed training data. Defaults to `./cleanDatasets`.
- **`training_labels`** (Optional): Path to training labels. Defaults to `./cleanDatasets`.
- **`testing_data`** (Optional): Path to preprocessed testing data. Defaults to `./cleanDatasets`.
- **`testing_labels`** (Optional): Path to testing labels. Defaults to `./cleanDatasets`.
- **`save_data_path`** (Optional): Directory for saving preprocessed training and testing data. Defaults to `./cleanDatasets`.
- **`save_labels_path`** (Optional): Directory for saving training and testing labels. Defaults to `./cleanDatasets`.
- **`results`** (Optional): Directory where evaluation results and model files will be stored.

#### **Cross-Validation (CV) and Model Name**
Defines settings for model training:

- **`cv`**: Specifies the number of cross-validation folds. Must be a positive integer (e.g., `cv: 5`).
- **`model_name`**: Specifies the machine learning model to be trained (e.g., `RandomForestRegressor`).

#### **Hyperparameters**
Defines hyperparameter ranges for the chosen model, allowing grid search or specific configurations.

### 2. Parsing Component

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

### 3. ETL Component (Preprocessing)

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

### 4. Transformation Logic

The user must specify their desired transformation logic for the system to apply in order to complete the ETL on the ingested dataset which the user has specified. At the moment, the system has one example of a specific transformation logic implemented for the Claifornia Housing Dataset.

---

### 5. ML Training Component

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

---

## How to Build and Run the System

The system supports multiple commands for running different phases of the pipeline. These commands should be executed in the terminal from the root directory of the project.

### 1. **Run the Full ETL and ML Pipeline**

To execute both the ETL and Machine Learning pipelines sequentially, use:

```bash
make all CONFIG_FILE=your_config.yaml
```
### 2. **Run Only the ETL Pipeline**

To preprocess the data using the ETL pipeline, use:
```bash
make etl CONFIG_FILE=your_config.yaml
```
### 3. **Run Only the ML Training Pipeline**

To train and evaluate the ML model, use:

```bash
make ml CONFIG_FILE=your_config.yaml
```
### 4. **Clean Up Generated Files**

To reset the workspace by removing all generated datasets, logs, and results, use:

```bash
make clean
```