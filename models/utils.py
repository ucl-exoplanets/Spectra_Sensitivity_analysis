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


def normalise(array, arr_max, arr_min):
    return (array - arr_min) / (arr_max - arr_min), arr_max, arr_min


def normalise_spectrum(spectrum):
    if spectrum.ndim == 1:
        spectrum = spectrum.reshape(-1, len(spectrum))
    local_min = spectrum.min(axis=1, keepdims=True)
    local_max = spectrum.max(axis=1, keepdims=True)
    norm_spectrum, _, _ = normalise(spectrum, local_max, local_min)
    return norm_spectrum, local_max, local_min
