# california_housing_transformation.py

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit
from base_data_transformation import BaseDataTransformation

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, 3] / X[:, 6]
        population_per_household = X[:, 5] / X[:, 6]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, 4] / X[:, 6]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

class CaliforniaHousingTransformation(BaseDataTransformation):
    def clean_data(self, data):
        """Performs stratified sampling and removes unnecessary columns specific to California housing."""
        # Create income categories for stratified sampling
        data["income_cat"] = pd.cut(
            data["median_income"],
            bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
            labels=[1, 2, 3, 4, 5]
        )
        
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(data, data["income_cat"]):
            strat_train_set = data.loc[train_index]
            strat_test_set = data.loc[test_index]

        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)
        
        self.housing = strat_train_set.drop("median_house_value", axis=1)
        self.housing_labels = strat_train_set["median_house_value"].copy()
        
        self.housing_test = strat_test_set.drop("median_house_value", axis=1)
        self.housing_labels_test = strat_test_set["median_house_value"].copy()
        
        return self.housing, self.housing_labels, self.housing_test, self.housing_labels_test

    def transform_features(self, data):
        """Applies California housing-specific transformations and feature scaling."""
        housing_num = data.drop("ocean_proximity", axis=1)
        
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler())
        ])
        
        num_attribs = list(housing_num)
        cat_attribs = ["ocean_proximity"]

        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs)
        ])
        
        housing_prepared = full_pipeline.fit_transform(data)
        return housing_prepared

