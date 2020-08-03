import numpy as np


def get_random_idx(array, portion=0.7, seed=42):
    np.random.seed(seed)
    return np.random.random_sample(len(array)) < portion


def standardise(data, mean, std):
    return (data - mean)/std


def project_back(data, mean, std):
    return data * std + mean


def random_index(data):
    index = np.arange(len(data))
    np.random.shuffle(index)
    return index
