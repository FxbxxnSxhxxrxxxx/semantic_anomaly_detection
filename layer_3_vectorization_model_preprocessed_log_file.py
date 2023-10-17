"""
Importing required modules and functions
"""
import pickle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

"""
Explanation:
The generate_vectorization_model function generates a vectorization model using Doc2Vec.

Inputs:
- ps_path_preprocessed_log_file: The path to the preprocessed log data file.
- ps_path_vectorization_model: The path where the vectorization model will be saved.
- ps_vector_size: The size of the vectors to be learned for each word and document.
- ps_window: The maximum distance between a target word and words around the target word.
- ps_min_count: The minimum frequency of words to be considered in the model.
- ps_epochs: The number of iterations over the corpus.
- ps_dbow_words: Whether to train word vectors in addition to document vectors (1) or not (0).
- ps_negative: The number of "noise words" to be drawn in negative sampling.

Outputs:
- Saves the trained Doc2Vec model to a file.
"""
def generate_vectorization_model(ps_path_preprocessed_log_file, ps_path_vectorization_model, ps_vector_size, ps_window, ps_min_count, ps_epochs, ps_dbow_words, ps_negative):
    # Open the preprocessed log file using pickle
    with open(f"{ps_path_preprocessed_log_file}.pkl", "rb") as file:
        # Load the preprocessed data into a variable called 'preprocessed_log_data'
        preprocessed_log_data = pickle.load(file)
    
    # Create a list of TaggedDocument objects, where each document is associated with a unique index
    tagged_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(preprocessed_log_data)]

    # Initialize a Doc2Vec model with the given parameters
    model = Doc2Vec(tagged_documents, vector_size=ps_vector_size, window=ps_window, min_count=ps_min_count, epochs=ps_epochs, dbow_words=ps_dbow_words, negative=ps_negative, workers=10)

    # Save the trained model to the specified path
    model.save(ps_path_vectorization_model)
