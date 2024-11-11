# errors.py

class ConfigError(Exception):
    """Exception raised for errors in the configuration file."""
    pass

class DataValidationError(Exception):
    """Exception raised for errors related to data validation."""
    pass

class DataValidationError(Exception):
    """Raised for errors related to data validation."""
    pass

class ModelInitializationError(Exception):
    """Raised for errors related to model initialization."""
    pass

class TrainingError(Exception):
    """Raised for errors during model training."""
    pass

class EvaluationError(Exception):
    """Raised for errors during model evaluation."""
    pass

class FileHandlingError(Exception):
    """Raised for errors related to file and directory handling."""
    pass
