"""
Importing required modules and functions
"""
import pandas as pd
import pickle
from gensim.models.doc2vec import Doc2Vec

"""
Explanation:
The vectorizing function vectorizes the log data using a previously saved Doc2Vec model.

Inputs:
- ps_path_vectorized_log_file: The path where the vectorized log data will be saved.
- ps_path_vectorization_model: The path to the previously saved Doc2Vec model.
- path_preprocessed_log_file: The path to the preprocessed log data file.
- ps_path_converted_log_file: The path to the converted log data file.

Outputs:
- Saves the vectorized log data to a CSV file.
"""
def vectorizing(ps_path_vectorized_log_file, ps_path_vectorization_model, path_preprocessed_log_file, ps_path_converted_log_file):
    # Load the previously saved Doc2Vec model
    model = Doc2Vec.load(ps_path_vectorization_model)

    # Open the preprocessed log file using pickle
    with open(f"{path_preprocessed_log_file}.pkl", "rb") as file:
        # Load the preprocessed data into a variable called 'preprocessed_log_data'
        preprocessed_log_data = pickle.load(file)

    # Read the converted log data from a CSV file into a pandas DataFrame
    converted_log_data = pd.read_csv(ps_path_converted_log_file)

    # Infer the sentence vectors for each preprocessed log entry using the Doc2Vec model
    sentence_vectors = [model.infer_vector(words) for words in preprocessed_log_data]

    # Concatenate the converted log data and the sentence vectors into a single DataFrame
    vectorized_log_data = pd.concat([converted_log_data, pd.DataFrame(sentence_vectors)], axis=1)

    # Save the vectorized log data to a CSV file
    vectorized_log_data.to_csv(ps_path_vectorized_log_file, index=False)
