"""
Importing required modules and functions
"""
import pandas as pd

"""
Explanation:
The conversion function converts a raw log file into a structured CSV file by splitting each log entry into neutral_info and study_info.

Inputs:
- ps_path_raw_log_file: The path to the raw log data file.
- ps_path_converted_log_file: The path where the converted log data will be saved.

Outputs:
- Saves the converted log data as a CSV file.
"""
def conversion(ps_path_raw_log_file, ps_path_converted_log_file):
    # Read the raw log file
    with open(ps_path_raw_log_file, "r", encoding='utf-8') as file:
        log_entries = file.readlines()

    # Initialize an empty list to store split log entries
    split_entries = []

    # Iterate through each log entry
    for entry in log_entries:
        # Split the entry into words
        words = entry.split()
        # Combine the first five words as neutral_info
        neutral_info = " ".join(words[:5])
        # Combine the remaining words as study_info
        study_info = " ".join(words[5:])
        # Append the split entry to the list
        split_entries.append([neutral_info, study_info])

    # Create a DataFrame from the split_entries list
    converted_log_data = pd.DataFrame(split_entries, columns=("neutral_info", "study_info"))

    # Save the DataFrame as a CSV file
    converted_log_data.to_csv(ps_path_converted_log_file, index=False)
