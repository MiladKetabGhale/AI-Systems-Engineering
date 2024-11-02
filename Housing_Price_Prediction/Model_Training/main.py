#!/usr/bin/env python
# coding: utf-8

import argparse
import pandas as pd
import numpy as np
import os
from housing_price_predictor import HousingPricePredictor

def main(config_file):
    # Initialize the predictor class
    predictor = HousingPricePredictor()

    # Load the configuration from the file
    config = predictor.load_config(config_file)

    # Dynamically set the paths for training and labels data
    dataset_name = os.path.basename(config_file).split('.')[0]  # Extracts 'california' from 'california.cfg'
    processed_data_dir = os.path.join("processed_data", f"{dataset_name}_housing")

    config['training_data_path'] = os.path.join(processed_data_dir, f"{dataset_name}_housing_prepared.csv")
    config['labels_data_path'] = os.path.join(processed_data_dir, f"{dataset_name}_housing_labels.csv")

    # Check if the paths exist
    if not os.path.exists(config['training_data_path']):
        raise FileNotFoundError(f"Training data file '{config['training_data_path']}' does not exist.")
    if not os.path.exists(config['labels_data_path']):
        raise FileNotFoundError(f"Labels data file '{config['labels_data_path']}' does not exist.")

    # Load the training data
    train_set = pd.read_csv(config['training_data_path'])
    train_labels = pd.read_csv(config['labels_data_path'])

    # Extract model name, CV folds, and hyperparameters from config
    model_name = config['model_name']
    cv_folds = config['cv_folds']
    hyperparameters = config['hyperparameters']

    # Fit and save the model based on the loaded config
    best_model, best_params, cv_results = predictor.fit_and_save(model_name, train_set, train_labels, hyperparameters, cv=cv_folds)

    # (Optional) Predict with the fitted model immediately
    # input_data = ...  # Load or prepare input data for predictions
    # predictions, errors = predictor.predict(input_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run housing price prediction model with config file.")
    parser.add_argument("config_file", help="Path to the configuration file")
    
    args = parser.parse_args()
    main(args.config_file)

