# ConfigLoader module with training_type determination

import os

def parse_value(value):
    """
    Parse a configuration value, converting it to a list of integers or floats if applicable.
    """
    try:
        # If the value contains commas, split into a list
        if ',' in value:
            return [parse_single_value(v.strip()) for v in value.split(',')]
        else:
            return parse_single_value(value)  # Return a single value
    except ValueError:
        return value  # Leave as a string if it's not a number

def parse_single_value(value):
    """
    Parse a single configuration value to its appropriate type.
    """
    value = value.strip().lower()  # Normalize the string
    if value in ['true', 'false']:
        return value == 'true'  # Convert to boolean
    try:
        return float(value) if '.' in value else int(value)  # Return as number
    except ValueError:
        return value  # Leave as string if conversion fails
    
def load_config(config_file_path):
    """Loads and parses configuration from a file, returning a dictionary."""
    config = {
        'cv': None,
        'model_name': None,
        'model_params': {},
        'training_data_path': None,
        'labels_data_path': None,
        'save_predictive_evaluations': False,
        'training_type': None  # Initialize training_type
    }

    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Configuration file '{config_file_path}' does not exist.")

    with open(config_file_path, 'r') as file:
        for line in file:
            line = line.split('#', 1)[0].strip()  # Remove comments
            if not line:
                continue
            
            key, value = line.split('=', 1)
            key, value = key.strip(), value.strip()

            # Parse key-value pairs
            if key == "cv":
                config['cv'] = int(value)
            elif key == "model":
                config['model_name'] = value.lower()
            elif key == "training_data_path":
                config['training_data_path'] = value
            elif key == "labels_data_path":
                config['labels_data_path'] = value
            elif key == "save_predictive_evaluations":
                config['save_predictive_evaluations'] = value.lower() == 'true'
            elif key.startswith("model_"):
                param_name = key[len("model_"):].strip()
                config['model_params'][param_name] = parse_value(value)

    # Determine training_type based on cv settings
    if config['cv'] > 1:
        config['training_type'] = "gridsearch"  # Grid search
    else:
        config['training_type'] = "simplefit"  # Simple fit without grid search
    # Validation check
    if not config['training_data_path'] or not config['labels_data_path']:
        raise ValueError("Both 'training_data_path' and 'labels_data_path' must be specified in the configuration file.")
    if config['cv'] is None:
        raise ValueError("'cv' must be specified in the configuration file.")
    if not config['model_name']:
        raise ValueError("If 'ensemble_training' is disabled, 'model_name' must be specified.")

    return config
