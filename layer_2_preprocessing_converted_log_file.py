"""
Importing required modules and functions
"""
import pandas as pd
import re
import pickle

"""
Explanation:
The preprocessing function preprocesses the log data by removing stop words, non-word characters, digits, and extra spaces.

Inputs:
- ps_path_converted_log_file: The path to the converted log data file.
- ps_path_preprocessed_log_file: The path where the preprocessed log data will be saved.

Outputs:
- Saves the preprocessed log data as a pickle file.
"""
def preprocessing(ps_path_converted_log_file, ps_path_preprocessed_log_file):
    
    # Define a list of stop words to be removed from the text
    stop_words = ['a', 'an', 'the', 'and', 'in', 'is', 'of', 'on', 'with']

    # Define a helper function to preprocess a single column of study info
    def sub_preprocessing(ps_column_study_info):
        # Ensure the input is a string
        ps_column_study_info = str(ps_column_study_info)

        # Remove non-word and non-space characters
        ps_column_study_info = re.sub(r"[^\w\s]", " ", ps_column_study_info)
        # Remove underscores and digits
        ps_column_study_info = re.sub(r"[_\d]", " ", ps_column_study_info)
        # Replace multiple spaces with a single space
        ps_column_study_info = re.sub(r"\s{2,}", " ", ps_column_study_info)
        # Convert the text to lowercase
        ps_column_study_info = ps_column_study_info.lower()
        # Split the text into words
        ps_column_study_info = ps_column_study_info.split()
        # Remove stop words from the list of words
        ps_column_study_info = [word for word in ps_column_study_info if word not in stop_words]
        # Return the preprocessed list of words
        return ps_column_study_info
    
    # Read the converted log file using pandas
    converted_log_data = pd.read_csv(ps_path_converted_log_file)
    
    # Initialize an empty list to store the preprocessed log data
    preprocessed_log_data = []
    
    # Iterate through each study_info entry in the converted log data
    for i in range(len(converted_log_data.study_info)):
        # Preprocess the study_info entry and append it to the preprocessed_log_data list
        preprocessed_log_data.append(sub_preprocessing(converted_log_data.study_info[i]))
    
    # Save the preprocessed log data as a pickle file
    with open(f"{ps_path_preprocessed_log_file}.pkl", "wb") as file:
        pickle.dump(preprocessed_log_data, file)
