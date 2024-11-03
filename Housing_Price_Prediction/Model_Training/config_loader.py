# config_loader.py

import os

def load_config(config_file_path, hyperparameters_grid):
    config = {
        'training_data_path': None,
        'labels_data_path': None,
        'cv_folds': 5,
        'model_name': None,
        'hyperparameters': {}
    }

    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Configuration file '{config_file_path}' does not exist.")

    with open(config_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, value = line.split('=', 1)
            key, value = key.strip(), value.strip()
            if key == "training_data_path":
                config['training_data_path'] = value
            elif key == "labels_data_path":
                config['labels_data_path'] = value
            elif key == "cv_folds":
                config['cv_folds'] = int(value)
            elif key == "model_name":
                config['model_name'] = value.lower()
            elif key.startswith("hyperparameter_"):
                param_name = key[len("hyperparameter_"):].strip()
                if value.isdigit():
                    config['hyperparameters'][param_name] = [int(value)]
                else:
                    try:
                        config['hyperparameters'][param_name] = [float(value)]
                    except ValueError:
                        config['hyperparameters'][param_name] = [value]

    if config['model_name'] in hyperparameters_grid:
        for param, default_values in hyperparameters_grid[config['model_name']].items():
            config['hyperparameters'].setdefault(param, default_values if isinstance(default_values, list) else [default_values])

    return config
