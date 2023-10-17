"""
Importing required modules and functions
"""
import os
from layer_1_converting_raw_log_file import conversion
from layer_2_preprocessing_converted_log_file import preprocessing
from layer_3_vectorization_model_preprocessed_log_file import generate_vectorization_model
from layer_4_vectorizing_preprocessed_log_file import vectorizing
from layer_5_anomaly_detection_model_vectorized_log_file import generate_anomaly_detection_model
from layer_6_detecting_anomalies_vectorized_log_file import detecting_anomalies

"""
Explanation:
The prepare_environment function prepares the environment by creating necessary directories for the anomaly detection process.

Inputs:
- None

Outputs:
- Creates necessary directories if they do not exist.
"""
def prepare_environment():
    # List of directories to be created
    paths=[
        "log_files/lake_raw_data/",
        "log_files/lake_conversion/",
        "log_files/lake_preprocessing/",
        "log_files/lake_vectorization_models/",
        "log_files/lake_vectorization/",
        "log_files/lake_anomaly_detection_models/",
        "log_files/lake_sorted_scores_train_log_data/",
        "log_files/lake_anomalies/",
    ]

    # Counter to keep track of the number of directories created
    counter=0
    for path in paths:
        directory=os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            counter+=1

    # If all directories are created, print success message and instruction to upload file
    if counter!=0:
        print("Directories have been created successfully.")
        print("Now: Upload the raw LogFile to the 'log_files/lake_raw_data' folder.")
    else:
        print("Directories already exist correctly.")

"""
Explanation:
The prepare_log_data function prepares the log data for further processing. It takes a raw log file, converts it from .txt to .csv format, and then preprocesses it.

Inputs:
- ps_file_name: The name of the raw data file.

Outputs:
- A converted and preprocessed data file.
"""
def prepare_log_data(ps_file_name):
    # Define the paths for the raw, converted, and preprocessed data files
    path_raw_log_file = f"log_files/lake_raw_data/{ps_file_name}"
    path_converted_log_file = f"log_files/lake_conversion/converted_{ps_file_name}"
    path_preprocessed_log_file = f"log_files/lake_preprocessing/preprocessed_{ps_file_name}"

    # Print the path of the raw data file
    print("Dataset That Shall Be Prepared.")
    print(f"Path: {path_raw_log_file}")

    # Convert the raw data file from .txt to .csv
    # The conversion function is imported from the layer_1_converting_raw_log_file module
    conversion(path_raw_log_file, path_converted_log_file)
    print("Conversion From .txt To .csv Done.")
    print(f"Path: {path_converted_log_file}")

    # Preprocess the converted data file
    # The preprocessing function is imported from the layer_2_preprocessing_converted_log_file module
    preprocessing(path_converted_log_file, path_preprocessed_log_file)
    print("Preprocessing Done.")
    print(f"Path: {path_preprocessed_log_file}")

"""
Explanation:
The fit function fits the model for anomaly detection using the provided preprocessed data file.

Inputs:
- ps_file_name: The name of the preprocessed data file.
- n_estimators (optional, default=100): The number of base estimators in the ensemble.
- vector_size (optional, default=100): The dimensionality of the feature vectors.
- window (optional, default=5): The maximum distance between the current and predicted word within a sentence.
- min_count (optional, default=5): The minimum number of word occurrences for it to be included in the vocabulary.
- epochs (optional, default=10): The number of iterations (epochs) over the corpus.
- dbow_words (optional, default=1): If set to 1, trains word vectors (in skip-gram fashion) simultaneous with DBOW.
- negative (optional, default=5): The number of “noise words” used for negative sampling.

Outputs:
- Generates necessary models for anomaly detection.
"""
def fit(ps_file_name, n_estimators=100, vector_size=300, window=15, min_count=5, epochs=20, dbow_words=1, negative=5):
    # Define the paths for each step of the process
    path_converted_log_file = f"log_files/lake_conversion/converted_{ps_file_name}"
    path_preprocessed_log_file = f"log_files/lake_preprocessing/preprocessed_{ps_file_name}"
    path_vectorized_train_log_file = f"log_files/lake_vectorization/vectorized_{ps_file_name}"
    path_vectorization_model = f"log_files/lake_vectorization_models/vectorization_model_{ps_file_name}"
    path_anomaly_detection_model = f"log_files/lake_anomaly_detection_models/anomaly_detection_model_{ps_file_name}"
    path_sorted_scores_train_log_data = f"log_files/lake_sorted_scores_train_log_data/sorted_scores_train_log_data_{ps_file_name}"

    # Print the path of the preprocessed data file
    print("Dataset That Shall Be Used For Generating Models.")
    print(f"Path: {path_converted_log_file}")

    # Generate a vectorization model using the preprocessed data
    generate_vectorization_model(path_preprocessed_log_file, path_vectorization_model, vector_size, window, min_count, epochs, dbow_words, negative)
    print("Generating Vectorization Model Done.")
    print(f"Path: {path_vectorization_model}")

    # Vectorize the preprocessed data using the generated vectorization model
    vectorizing(path_vectorized_train_log_file, path_vectorization_model, path_preprocessed_log_file, path_converted_log_file)
    print("Vectorization Done.")
    print(f"Path: {path_vectorized_train_log_file}")

    # Generate an anomaly detection model using the vectorized data
    generate_anomaly_detection_model(path_vectorized_train_log_file, path_anomaly_detection_model, path_sorted_scores_train_log_data, n_estimators)
    print("Generating Anomaly Detection Model Done.")
    print(f"Path: {path_anomaly_detection_model}")

"""
Explanation:
The apply_anomaly_detection function is a high-level function that performs the entire anomaly detection process, including vectorization and anomaly detection. 
It takes a preprocessed data file as input and outputs a report containing the detected anomalies in their respective folders.

Inputs:
- ps_file_name: The name of the preprocessed data file.
- percentage_threshold (optional, default=1): The proportion of outliers (Input in %) in the representative training data set the model is based on.
  The threshold is based on the training data set.
- model (optional, default=train data of model): The name of the model to be used for vectorization and anomaly detection. If not provided, the file name will be used as the model name. 
  By doing so, the function assumes that a model already exists but which is based on the same data that is about to be analyzed.

Outputs:
- A report containing the detected anomalies is saved in their respective folders.
"""
def apply_anomaly_detection(ps_file_name, percentage_threshold=5, model=None):
    # If no model name is provided, use the file name as the model name. By doing so, the function assumes 
    # that a model already exists but which is based on the same data that is about to be analyzed.
    if model == None:
        model = ps_file_name

    # Define the paths for each step of the process
    path_converted_log_file = f"log_files/lake_conversion/converted_{ps_file_name}"
    path_preprocessed_log_file = f"log_files/lake_preprocessing/preprocessed_{ps_file_name}"
    path_vectorized_test_log_file = f"log_files/lake_vectorization/vectorized_{ps_file_name}"
    path_anomalies_detected_log_file = f"log_files/lake_anomalies/anomalies_detected_{ps_file_name}"
    path_vectorization_model = f"log_files/lake_vectorization_models/vectorization_model_{model}"
    path_anomaly_detection_model = f"log_files/lake_anomaly_detection_models/anomaly_detection_model_{model}"
    path_sorted_scores_train_log_data = f"log_files/lake_sorted_scores_train_log_data/sorted_scores_train_log_data_{model}"

    # Print the path of the raw data file
    print("Dataset That Shall Be Analyzed.")
    print(f"Path: {path_converted_log_file}")

    # Vectorize the preprocessed data using the specified model
    vectorizing(path_vectorized_test_log_file, path_vectorization_model, path_preprocessed_log_file, path_converted_log_file)
    print("Vectorization Done.")
    print(f"Path: {path_vectorized_test_log_file}")

    # Detect anomalies in the vectorized data using the specified model
    detecting_anomalies(path_sorted_scores_train_log_data, path_vectorized_test_log_file, path_anomaly_detection_model, path_anomalies_detected_log_file, percentage_threshold/100)
    print("Analysis Done.")
    print(f"Path: {path_anomalies_detected_log_file}")
