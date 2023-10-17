""" 
Importing required modules and functions
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

"""
Explanation:
The generate_anomaly_detection_model function generates an anomaly detection model using Isolation Forest.

Inputs:
- ps_path_vectorized_train_log_file: The path to the vectorized train log data file.
- ps_path_anomaly_detection_model: The path where the anomaly detection model will be saved.
- ps_path_sorted_scores_train_log_data: The path where the sorted anomaly scores of the train log data will be saved.
- ps_n_estimators: The number of base estimators in the ensemble.

Outputs:
- Saves the trained Isolation Forest model to a file.
- Saves the sorted anomaly scores of the train log data to a file.
"""
def generate_anomaly_detection_model(ps_path_vectorized_train_log_file, ps_path_anomaly_detection_model, ps_path_sorted_scores_train_log_data, ps_n_estimators):
    # Read the vectorized train log data from a CSV file into a pandas DataFrame
    vectorized_train_log_file = pd.read_csv(ps_path_vectorized_train_log_file)

    # Initialize the Isolation Forest model with the specified number of base estimators
    anomaly_detection_model = IsolationForest(n_estimators=ps_n_estimators, n_jobs=-1)

    # Fit the model to the vectorized train log data, excluding 'neutral_info' and 'study_info' columns
    anomaly_detection_model.fit(vectorized_train_log_file.drop(['neutral_info', 'study_info'], axis=1))

    # Calculate anomaly scores for the train log data
    scores_train_log_data = anomaly_detection_model.decision_function(vectorized_train_log_file.drop(['neutral_info','study_info'], axis=1))

    # Sort the anomaly scores in ascending order
    sorted_scores_train_log_data = np.sort(scores_train_log_data)

    # Save the sorted anomaly scores of the train log data to a file using pickle
    with open(f"{ps_path_sorted_scores_train_log_data}.pkl", "wb") as file:
        pickle.dump(sorted_scores_train_log_data, file)

    # Save the trained Isolation Forest model to a file using pickle
    with open(f"{ps_path_anomaly_detection_model}.pkl", "wb") as file:
        pickle.dump(anomaly_detection_model, file)
