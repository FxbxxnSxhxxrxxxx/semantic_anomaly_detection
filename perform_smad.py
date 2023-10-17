#!/home/schlarma/anaconda3/bin/python

# Import anomaly detection functions
from log_entries_anomaly_detection import prepare_log_data, apply_anomaly_detection

# Get user inputs: log file, model, and percentage threshold
user_input_file = input("Enter the raw log file to analyze: ")
user_input_model = input("Enter the trained model name: ")
user_input_percentage_threshold = float(input("Enter the percentage threshold: "))

# Process log data
prepare_log_data(user_input_file)

# Apply anomaly detection with user-defined parameters
apply_anomaly_detection(user_input_file, user_input_percentage_threshold, user_input_model)
