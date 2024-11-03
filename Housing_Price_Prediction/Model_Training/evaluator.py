# evaluator.py

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, input_data, true_labels=None, compute_errors=True):
    print("Generating predictions...")
    predictions = model.predict(input_data)

    evaluation_metrics = {}
    if compute_errors and true_labels is not None:
        evaluation_metrics['mean_squared_error'] = mean_squared_error(true_labels, predictions)
        evaluation_metrics['root_mean_squared_error'] = np.sqrt(mean_squared_error(true_labels, predictions))
        evaluation_metrics['mean_absolute_error'] = mean_absolute_error(true_labels, predictions)
        evaluation_metrics['r2_score'] = r2_score(true_labels, predictions)

    return predictions, evaluation_metrics if evaluation_metrics else None
