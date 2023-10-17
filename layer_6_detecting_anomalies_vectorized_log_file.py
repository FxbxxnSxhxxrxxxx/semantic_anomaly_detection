"""
Importing required modules and functions
"""
import pandas as pd
import pickle
import numpy as np
import copy

"""
Explanation:
The function 'detecting_anomalies' is used to detect anomalies in a given dataset using a pre-trained anomaly detection model.

Inputs:
- ps_path_sorted_scores_train_log_data: The path to the sorted anomaly scores of the train log data.
- ps_path_vectorized_test_log_file: The path to the vectorized test log file which is about to be analyzed.
- ps_path_anomaly_detection_model: The path to the anomaly detection model.
- ps_path_anomalies_detected_log_file: The path where the detected anomalies will be saved.
- ps_percentage_threshold: The proportion of outliers (Input in %) in the representative training data set the model is based on.
  The threshold is based on the training data set.

Outputs:
- A CSV file containing the detected anomalies is saved at the specified path.
"""
def detecting_anomalies(ps_path_sorted_scores_train_log_data, ps_path_vectorized_test_log_file, ps_path_anomaly_detection_model, ps_path_anomalies_detected_log_file, ps_percentage_threshold):
    # Load the anomaly detection model from a pickle file
    with open(f"{ps_path_anomaly_detection_model}.pkl", "rb") as file:
        anomaly_detection_model = pickle.load(file)
        
    # Load the vectorized test log data from a CSV file into a pandas DataFrame
    vectorized_test_log_data = pd.read_csv(ps_path_vectorized_test_log_file)
    
    # Use the anomaly detection model to compute anomaly scores for the test log data
    # Exclude 'neutral_info' and 'study_info' columns before computing the scores
    scores_test_log_data = anomaly_detection_model.decision_function(vectorized_test_log_data.drop(['neutral_info','study_info'], axis=1))
    
    # Load the sorted anomaly scores of the train log data from a pickle file
    with open(f"{ps_path_sorted_scores_train_log_data}.pkl", "rb") as file:
        sorted_scores_train_log_data = pickle.load(file)

    # Calculate the threshold using the train log data
    # The threshold is calculated as the score at the position of the proportion of contamination in the sorted scores
    threshold = sorted_scores_train_log_data[int((ps_percentage_threshold)*len(sorted_scores_train_log_data))]
    
    # Classify the data points in the test log data as anomalies or normal based on the threshold
    # If the score is less than the threshold, it is classified as an anomaly (-1), otherwise it is classified as normal (1)
    predictions = np.where(scores_test_log_data < threshold, -1, 1)
    
    # Create a copy of the vectorized test log data and add a new column for the anomaly predictions
    anomalies_detected_test_log_data = copy.deepcopy(vectorized_test_log_data)
    anomalies_detected_test_log_data['anomaly'] = predictions

    # Filter the data to only include rows where the 'anomaly' column is equal to -1 (anomalies)
    anomalies_detected_test_log_data = anomalies_detected_test_log_data[anomalies_detected_test_log_data['anomaly'] == -1]
    
    # Drop the 'anomaly' column before saving the data as CSV
    # This is done to only include the original data in the output file
    anomalies_detected_test_log_data[['neutral_info', 'study_info']].to_csv(ps_path_anomalies_detected_log_file, index=False)
