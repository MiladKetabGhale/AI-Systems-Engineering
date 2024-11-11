import yaml
import logging
from sklearn.utils import all_estimators
from errors import ConfigError, DataValidationError

def get_model_class(model_name):
    """
    Dynamically retrieves the model class for the specified model name.
    """
    # Retrieve all available estimators in scikit-learn
    all_models = {name: clazz for name, clazz in all_estimators()}

    # Check if the specified model is available
    if model_name not in all_models:
        available_models = ", ".join(all_models.keys())
        raise ConfigError("Model '{}' is not recognized. Available models are: {}".format(model_name, available_models))

    # Get the model class directly from the dictionary
    return all_models[model_name]

def validate_hyperparameters(model_name, hyperparameters):
    """
    Validates the hyperparameters based on the selected model's allowed parameters using sklearn's get_params().
    """
    model_class = get_model_class(model_name)
    model = model_class()  # Initialize model instance to get its parameters
    
    allowed_params = model.get_params()  # Retrieve allowed parameters
    
    for param, value in hyperparameters.items():
        # Check if the hyperparameter is valid for this model
        if param not in allowed_params:
            raise ConfigError(
                "Invalid hyperparameter '{}' for model '{}'. Allowed parameters: {}".format(
                    param, model_name, ", ".join(allowed_params.keys())
                )
            )

        # Check type compatibility: allow NoneType for parameters that can be None
        allowed_type = type(allowed_params[param])
        if allowed_params[param] is None:
            allowed_type = (type(None), int, float, str)

        # Validate each parameter type; allow lists for GridSearchCV
        if not isinstance(value, (allowed_type, list)):
            raise ConfigError(
                "Hyperparameter '{}' should be of type {} for model '{}'. Received type: {}".format(
                    param, allowed_type, model_name, type(value)
                )
            )

def parse_config(yaml_path):
    """
    Parses and validates the YAML configuration file for model training.
    
    Args:
    - yaml_path (str): Path to the YAML configuration file.
    
    Returns:
    - parsed_config (dict): Dictionary containing parsed configuration suitable for model training and GridSearchCV.
    """
    try:
        logging.info("Starting configuration parsing...")

        # Attempt to load the configuration file
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info("Configuration file loaded successfully.")

    except FileNotFoundError:
        logging.critical("Configuration file not found: {}".format(yaml_path))
        raise ConfigError("The specified configuration file was not found.")
    
    except yaml.YAMLError:
        logging.critical("Error parsing YAML configuration file.")
        raise ConfigError("There was an error parsing the configuration file.")

    # Initialize the parsed configuration dictionary
    parsed_config = {
        "paths": {},
        "model_name": None,
        "cv": None,
        "param_grid": {},
        "evaluation_metric": None
    }

    # Parse and validate paths
    try:
        logging.info("Parsing paths...")
        paths = config.get("paths", {})
        parsed_config["paths"]["training_data"] = paths.get("training_data")
        parsed_config["paths"]["training_labels"] = paths.get("training_labels")
        parsed_config["paths"]["results"] = paths.get("results", None)  # Optional field

        # Check mandatory paths
        if not parsed_config["paths"]["training_data"]:
            logging.error("Missing mandatory path: 'training_data' is required.")
            raise ConfigError("Missing mandatory path: 'training_data' is required.")
        if not parsed_config["paths"]["training_labels"]:
            logging.error("Missing mandatory path: 'training_labels' is required.")
            raise ConfigError("Missing mandatory path: 'training_labels' is required.")
        logging.info("Paths parsed successfully.")

    except KeyError:
        logging.critical("Configuration file is missing necessary paths.")
        raise ConfigError("The configuration file is missing one or more mandatory paths.")
    
    # Parse and validate model name
    try:
        logging.info("Parsing model configuration...")
        model_names = config.get("model_config", {}).get("model_name", [])
        uncommented_models = [name for name in model_names if name]
        
        if len(uncommented_models) == 0:
            logging.error("No model selected. Please uncomment one model in the 'model_name' section.")
            raise ConfigError("No model selected. Please uncomment one model in the 'model_name' section.")
        elif len(uncommented_models) > 1:
            logging.error("Multiple models selected. Please select only one model in the 'model_name' section.")
            raise ConfigError("Multiple models selected. Please select only one model in the 'model_name' section.")
        else:
            parsed_config["model_name"] = uncommented_models[0]
        logging.info("Model configuration parsed successfully.")

    except KeyError:
        logging.critical("Configuration file is missing the 'model_name' field.")
        raise ConfigError("The 'model_name' field is required in the configuration file.")

    # Parse cross-validation (cv) value
    try:
        logging.info("Parsing cross-validation settings...")
        cv_value = config.get("model_config", {}).get("cv")
        if cv_value is None:
            logging.error("Missing mandatory field: 'cv' is required.")
            raise ConfigError("Missing mandatory field: 'cv' is required.")
        elif not isinstance(cv_value, int) or cv_value < 1:
            logging.error("'cv' must be a positive integer starting from 1.")
            raise ConfigError("'cv' must be a positive integer starting from 1.")
        parsed_config["cv"] = cv_value
        logging.info("Cross-validation setting parsed successfully.")

    except KeyError:
        logging.critical("Configuration file is missing the 'cv' field.")
        raise ConfigError("The 'cv' field is required in the configuration file.")
    
    # Parse evaluation metric
    try:
        logging.info("Parsing evaluation metrics...")
        evaluation_metric = config.get("model_config", {}).get("evaluation_metric")
        if not evaluation_metric or not isinstance(evaluation_metric, list):
            logging.error("Missing mandatory field: 'evaluation_metric' must be a non-empty list of strings.")
            raise ConfigError("Missing mandatory field: 'evaluation_metric' must be a non-empty list of strings.")

        # Validate that each metric is a non-empty string
        for metric in evaluation_metric:
            if not isinstance(metric, str) or not metric.strip():
                logging.error("Each item in 'evaluation_metric' must be a non-empty string.")
                raise ConfigError("Each item in 'evaluation_metric' must be a non-empty string.")
        parsed_config["evaluation_metric"] = evaluation_metric
        logging.info("Evaluation metrics parsed successfully.")

    except KeyError:
        logging.critical("Configuration file is missing the 'evaluation_metric' field.")
        raise ConfigError("The 'evaluation_metric' field is required in the configuration file.")
    
    # Parse and validate hyperparameters
    try:
        logging.info("Parsing hyperparameters...")
        model_hyperparameters = config.get("model_config", {}).get("model_hyperparameters", {})
        selected_model_params = model_hyperparameters.get(parsed_config["model_name"], {})

        # Validate hyperparameters
        validate_hyperparameters(parsed_config["model_name"], selected_model_params)

        # Format hyperparameters for GridSearchCV
        param_grid = {}
        for param, value in selected_model_params.items():
            if isinstance(value, list):
                param_grid[param] = value
            elif value is not None:
                param_grid[param] = [value]  # Single value wrapped in a list
        parsed_config["param_grid"] = param_grid
        print("*********Hyperparameters parsed and validated successfully.**************")
        logging.info("Hyperparameters parsed and validated successfully.")

    except KeyError:
        logging.critical("Error parsing or validating hyperparameters.")
        raise ConfigError("There was an error parsing or validating the hyperparameters in the configuration file.")
    print("Configuration parsing completed successfully.")
    logging.info("Configuration parsing completed successfully.")
    return parsed_config

# Example usage
if __name__ == "__main__":
    try:
        config_path = 'config.yaml'  # Path to your YAML config file
        parsed_data = parse_config(config_path)
        print(parsed_data)
    except ConfigError as e:
        print("Configuration Error: {}".format(e))

