import os
import tarfile
import urllib.parse
import urllib.request
import pandas as pd

class DataIngestion:
    def __init__(self, data_url, data_path):
        self.data_url = data_url
        self.data_path = data_path

    def download_data(self):
        """Downloads the dataset if it's a URL; otherwise, uses a local path."""
        if urllib.parse.urlparse(self.data_url).scheme in ('http', 'https'):
            # Download from URL
            if not os.path.isdir(self.data_path):
                os.makedirs(self.data_path)
            tgz_path = os.path.join(self.data_path, "housing.tgz")
            urllib.request.urlretrieve(self.data_url, tgz_path)

            with tarfile.open(tgz_path) as housing_tgz:
                housing_tgz.extractall(path=self.data_path)
        elif os.path.isfile(self.data_url):
            # Handle local file directly
            print(f"Using local data file: {self.data_url}")
            # Extract if necessary or copy to the data path
            if not os.path.isdir(self.data_path):
                os.makedirs(self.data_path)
            if self.data_url.endswith('.tgz'):
                with tarfile.open(self.data_url) as housing_tgz:
                    housing_tgz.extractall(path=self.data_path)
            else:
                raise ValueError("Unsupported local file format. Only .tgz files are supported.")

    def load_data(self):
        """Loads the housing data into a DataFrame."""
        csv_path = os.path.join(self.data_path, "housing.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
        return pd.read_csv(csv_path)

