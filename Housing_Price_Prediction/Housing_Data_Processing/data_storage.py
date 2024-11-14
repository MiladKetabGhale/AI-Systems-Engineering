# data_storage.py

import os
import numpy as np
import pandas as pd

class DataStorage:
    def __init__(self, save_path='./cleanDatasets', dataset_name="default"):
        self.save_path = save_path
        self.dataset_name = dataset_name
        os.makedirs(self.save_path, exist_ok=True)  # Ensure save_path exists

    def save_labels(self, labels):
        """Saves training labels to a CSV file."""
        labels_path = os.path.join(self.save_path, f"{self.dataset_name}_labels.csv")
        np.savetxt(labels_path, labels, delimiter=",")
        print(f"Labels saved at {labels_path}")

    def save_transformed_data(self, data):
        """Saves transformed training data to a CSV file."""
        data_path = os.path.join(self.save_path, f"{self.dataset_name}_prepared.csv")
        np.savetxt(data_path, data, delimiter=",")
        print(f"Processed data saved at {data_path}")

    def save_test_data(self, test_data, test_labels):
        """Saves testing data and labels to CSV files."""
        test_data_path = os.path.join(self.save_path, f"{self.dataset_name}_test.csv")
        test_labels_path = os.path.join(self.save_path, f"{self.dataset_name}_test_labels.csv")

        pd.DataFrame(test_data).to_csv(test_data_path, index=False)
        pd.DataFrame(test_labels).to_csv(test_labels_path, index=False)

        print(f"Test data saved at {test_data_path}")
        print(f"Test labels saved at {test_labels_path}")

