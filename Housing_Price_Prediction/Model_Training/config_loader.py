# ConfigLoader module with training_type determination

import os

def parse_value(value):
    """Parses individual configuration values, handling lists if needed."""
    if ',' in value:
        return [parse_value(v.strip()) for v in value.split(',')]
    try:
        return float(value) if '.' in value else int(value)
    except ValueError:
        return value

def load_config(config_file_path):
    """Loads and parses configuration from a file, returning a dictionary."""
    config = {
        'cv': None,
        'model_name': None,
        'ensemble_training': False,
        'ensemble_name': None,
        'ensemble_base_model': None,
        'ensemble_params': {},
        'base_params': {},
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
            elif key == "model_name":
                config['model_name'] = value.lower()
            elif key == "ensemble_training":
                config['ensemble_training'] = value.lower() == 'true'
            elif key == "ensemble_name":
                config['ensemble_name'] = value.lower()
            elif key == "ensemble_base_model":
                config['ensemble_base_model'] = value.lower()
            elif key == "training_data_path":
                config['training_data_path'] = value
            elif key == "labels_data_path":
                config['labels_data_path'] = value
            elif key == "save_predictive_evaluations":
                config['save_predictive_evaluations'] = value.lower() == 'true'
            elif key.startswith("ensemble_"):
                param_name = key[len("ensemble_"):].strip()
                config['ensemble_params'][param_name] = parse_value(value)
            elif key.startswith("base_model_hyperparameters_"):
                param_name = key[len("base_model_hyperparameters_"):].strip()
                config['base_params'][param_name] = parse_value(value)

    # Determine training_type based on cv and ensemble settings
    if config['ensemble_training']:
        config['training_type'] = "ensemble"  # Ensemble training
    elif config['cv'] > 1:
        config['training_type'] = "gridsearch"  # Grid search
    else:
        config['training_type'] = "simplefit"  # Simple fit without grid search

    # Debugging output to confirm all expected keys are present
    print("Config loaded with keys:", config.keys())
    print("Config content:", config)

    # Validation check
    if not config['training_data_path'] or not config['labels_data_path']:
        raise ValueError("Both 'training_data_path' and 'labels_data_path' must be specified in the configuration file.")
    if config['cv'] is None:
        raise ValueError("'cv' must be specified in the configuration file.")
    if config['ensemble_training']:
        if not config['ensemble_name'] or not config['ensemble_base_model']:
            raise ValueError("If 'ensemble_training' is enabled, 'ensemble_name' and 'ensemble_base_model' must be specified.")
    else:
        if not config['model_name']:
            raise ValueError("If 'ensemble_training' is disabled, 'model_name' must be specified.")

    return config
