#!/usr/bin/env python
# coding: utf-8

import argparse
import pandas as pd
import numpy as np
import os
from config_loader import load_config
from model_initializer import initialize_model, model_mapping, hyperparameters_grid
from training import train_model
from model_saver import save_model
from evaluator import evaluate_model

def main(config_file):
    # Load configuration
    config = load_config(config_file, hyperparameters_grid)

    # Get paths for training and labels data from the configuration
    training_data_path = config['training_data_path']
    labels_data_path = config['labels_data_path']

    # Check if the paths exist
    if not os.path.exists(training_data_path):
        raise FileNotFoundError(f"Training data file '{training_data_path}' does not exist.")
    if not os.path.exists(labels_data_path):
        raise FileNotFoundError(f"Labels data file '{labels_data_path}' does not exist.")

    # Load the training data
    train_set = pd.read_csv(training_data_path)
    train_labels = pd.read_csv(labels_data_path).values.ravel()  # Convert DataFrame to 1D array

    # Initialize the model with hyperparameters
    model = initialize_model(config['model_name'], model_params=config.get('hyperparameters'))

    # Train the model
    trained_model = train_model(model, train_set, train_labels, param_grid=config.get('hyperparameters'), cv=config['cv_folds'])

    # Save the trained model
    model_save_path = os.path.join("saved_model_params_cvres", f"{config['model_name']}_model.pkl")
    save_model(trained_model, config.get('hyperparameters'), None, config['model_name'], dir_path="saved_model_params_cvres")

    # Print a message indicating successful training and saving
    print(f"Model training and saving completed successfully. Model saved at '{model_save_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run housing price prediction model with config file.")
    parser.add_argument("config_file", help="Path to the configuration file")
    
    args = parser.parse_args()
    main(args.config_file)

