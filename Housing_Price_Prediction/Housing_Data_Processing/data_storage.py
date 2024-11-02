# data_storage.py

# data_storage.py

import os
import numpy as np

class DataStorage:
    def __init__(self, save_path='./cleanDatasets'):
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)  # Ensure save_path exists

    def save_labels(self, labels):
        labels_path = os.path.join(self.save_path, "california_housing_labels.csv")
        np.savetxt(labels_path, labels, delimiter=",")
        print(f"Labels saved at {labels_path}")

    def save_transformed_data(self, data):
        data_path = os.path.join(self.save_path, "california_housing_prepared.csv")
        np.savetxt(data_path, data, delimiter=",")
        print(f"Processed data saved at {data_path}")

