import numpy as np
from scipy.spatial.distance import hamming, cosine


def hamming_distance(training_set_vectors, query_vector, top_n=20):
    '''
    Calculates hamming distance between query image and all training set images
    
    :param training_set_vectors: numpy matrix
    :param query_vector: numpy vector
    :param top_n: integer
    '''
    distances = []
    
    for training_vector in training_set_vectors:
        distances.append(hamming(training_vector, query_vector))
    return np.argsort(distances)[:top_n]

def cosine_distance(training_set_vectors, query_vector, top_n=20):
    '''
    Calculates cosine distance between query image and all training set images
    
    :param training_set_vectors: numpy matrix
    :param query_vector: numpy vector
    :param top_n: integer
    '''
    
    distances = []
    
    for training_vector in training_set_vectors:
        distances.append(cosine(training_vector, query_vector[0]))
    return np.argsort(distances)[:top_n]