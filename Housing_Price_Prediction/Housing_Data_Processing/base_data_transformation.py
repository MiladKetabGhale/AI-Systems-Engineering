# base_data_transformation.py

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

class BaseDataTransformation(ABC):
    def __init__(self, save_path='./cleanDatasets'):
        self.save_path = save_path
        self.housing = None
        self.housing_labels = None

    @abstractmethod
    def clean_data(self, data: pd.DataFrame):
        """Abstract method for data cleaning."""
        pass

    @abstractmethod
    def transform_features(self, data: pd.DataFrame):
        """Abstract method for feature transformation."""
        pass

    def set_save_path(self, path):
        """Sets a new path for saving transformed data and labels."""
        self.save_path = path

