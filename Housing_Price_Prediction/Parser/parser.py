import yaml
import logging
from sklearn.utils import all_estimators
from errors import ConfigError, DataValidationError

def get_model_class(model_name):
    """
    Dynamically retrieves the model class for the specified model name.
    """
    all_models = {name: clazz for name, clazz in all_estimators()}
    if model_name not in all_models:
        available_models = ", ".join(all_models.keys())
        raise ConfigError(f"Model '{model_name}' is not recognized. Available models are: {available_models}")
    return all_models[model_name]

def validate_hyperparameters(model_name, hyperparameters):
    model_class = get_model_class(model_name)
    model = model_class()
    allowed_params = model.get_params()
    
    for param, value in hyperparameters.items():
        if param not in allowed_params:
            raise ConfigError(
                f"Invalid hyperparameter '{param}' for model '{model_name}'. Allowed parameters: {', '.join(allowed_params.keys())}"
            )
        allowed_type = type(allowed_params[param]) if allowed_params[param] is not None else (type(None), int, float, str)
        if not isinstance(value, (allowed_type, list)):
            raise ConfigError(
                f"Hyperparameter '{param}' should be of type {allowed_type} for model '{model_name}'. Received type: {type(value)}"
            )

def parse_config(yaml_path):
    """
    Parses and validates the YAML configuration file for model training.
    """
    try:
        logging.info("Starting configuration parsing...")
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info("Configuration file loaded successfully.")
    except FileNotFoundError:
        raise ConfigError("The specified configuration file was not found.")
    except yaml.YAMLError:
        raise ConfigError("There was an error parsing the configuration file.")

    parsed_config = {
        "paths": {},
        "model_name": None,
        "cv": None,
        "param_grid": {},
        "evaluation_metric": None
    }

    try:
        logging.info("Parsing paths...")
        paths = config.get("paths", {})
        
        parsed_config["paths"]["training_data"] = paths.get("training_data", "")
        parsed_config["paths"]["training_labels"] = paths.get("training_labels", "")
        parsed_config["paths"]["testing_data"] = paths.get("testing_data", "")
        parsed_config["paths"]["testing_labels"] = paths.get("testing_labels", "")
        parsed_config["paths"]["results"] = paths.get("results", None)
        parsed_config["paths"]["transformation_logic_path"] = paths.get("transformation_logic_path")
        parsed_config["paths"]["data_ingestion_source"] = paths.get("data_ingestion_source")
        
        if not parsed_config["paths"]["data_ingestion_source"]:
            raise ConfigError("The 'data_ingestion_source' must be specified in the 'paths' section of the config file.")

        logging.info("Paths parsed successfully.")
    except KeyError:
        raise ConfigError("The configuration file is missing one or more mandatory paths.")

    try:
        logging.info("Parsing model configuration...")
        model_names = config.get("model_config", {}).get("model_name", [])
        uncommented_models = [name for name in model_names if name]
        
        if len(uncommented_models) == 0:
            raise ConfigError("No model selected. Please uncomment one model in the 'model_name' section.")
        elif len(uncommented_models) > 1:
            raise ConfigError("Multiple models selected. Please select only one model in the 'model_name' section.")
        else:
            parsed_config["model_name"] = uncommented_models[0]
        logging.info("Model configuration parsed successfully.")
    except KeyError:
        raise ConfigError("The 'model_name' field is required in the configuration file.")

    try:
        logging.info("Parsing cross-validation settings...")
        cv_value = config.get("model_config", {}).get("cv")
        if cv_value is None or not isinstance(cv_value, int) or cv_value < 1:
            raise ConfigError("'cv' must be a positive integer starting from 1.")
        parsed_config["cv"] = cv_value
        logging.info("Cross-validation setting parsed successfully.")
    except KeyError:
        raise ConfigError("The 'cv' field is required in the configuration file.")

    try:
        logging.info("Parsing evaluation metrics...")
        evaluation_metric = config.get("model_config", {}).get("evaluation_metric")
        if not evaluation_metric or not isinstance(evaluation_metric, list):
            raise ConfigError("The 'evaluation_metric' must be a non-empty list of strings.")
        for metric in evaluation_metric:
            if not isinstance(metric, str) or not metric.strip():
                raise ConfigError("Each item in 'evaluation_metric' must be a non-empty string.")
        parsed_config["evaluation_metric"] = evaluation_metric
        logging.info("Evaluation metrics parsed successfully.")
    except KeyError:
        raise ConfigError("The 'evaluation_metric' field is required in the configuration file.")

    try:
        logging.info("Parsing hyperparameters...")
        model_hyperparameters = config.get("model_config", {}).get("model_hyperparameters", {})
        selected_model_params = model_hyperparameters.get(parsed_config["model_name"], {})

        validate_hyperparameters(parsed_config["model_name"], selected_model_params)
        
        param_grid = {}
        for param, value in selected_model_params.items():
            param_grid[param] = value if isinstance(value, list) else [value]
        parsed_config["param_grid"] = param_grid
        logging.info("Hyperparameters parsed and validated successfully.")
    except KeyError:
        raise ConfigError("Error parsing or validating hyperparameters in the configuration file.")

    logging.info("Configuration parsing completed successfully.")
    return parsed_config

