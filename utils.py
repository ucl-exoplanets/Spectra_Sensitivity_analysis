import numpy as np
import glob
import pandas as pd


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


def shuffle_spectrum(spectrum, error, times=10):
    # pre allocate space
    shuffled_batch = np.zeros((spectrum.shape[0]*times, spectrum.shape[1]))
    for spec_idx, one_spec in enumerate(spectrum):
        for idx in range(times):
            shuffled_spectrum = np.random.normal(
                loc=one_spec, scale=error[spec_idx])
            shuffled_batch[spec_idx*times+idx, :] = shuffled_spectrum
    return shuffled_batch


def load_history(checkpoint_dir):
    print(checkpoint_dir)
    training_log = glob.glob(checkpoint_dir + "history/*.log")
    print(training_log)
    # TODO return none when list is empty
    history_data = pd.read_csv(training_log[0])
    epoch = len(history_data['loss'])

    training_loss = np.zeros((len(training_log), epoch))
    valid_loss = np.zeros((len(training_log), epoch))
    for idx, hist in enumerate(training_log):
        history_data = pd.read_csv(hist)
        training_loss[idx] += history_data['loss'][:]
        valid_loss[idx] += history_data['val_loss'][:]
    return training_loss, valid_loss
