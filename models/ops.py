import numpy as np
import pandas as pd
import glob
from .constants import label, R_J
from .utils import standardise


def preprocessing(spectrum_file, param_file):
    # Only the first 10 items are trainable.
    trainable_param = param_file.values[:, :9]
    # turn into logarithm.
    trainable_param[:, :6] = np.log10(trainable_param[:, :6])
    trainable_param[:, -1] = np.log10(trainable_param[:, -1])
    # separate into spectrum, error and wl
    spectrum = spectrum_file[:, 0, :]
    error = spectrum_file[:, 1, :]
    wl = spectrum_file[0, 2, :]
    return spectrum, error, wl, trainable_param


def compute_MSE(y_true, y_pred, gas_label=None):
    MSE = np.mean(np.square(y_true-y_pred), axis=0)
    length = len(MSE)
    MSE = MSE.reshape(1, -1)
    if gas_label is None:
        gas_label = label
    df = pd.DataFrame(MSE, columns=label[:length])
    df['Mean MSE'] = np.mean(np.square(y_true-y_pred))
    return df


def shuffle_spectrum(spectrum, error, times=10):
    # pre allocate space
    shuffled_batch = np.zeros((spectrum.shape[0]*times, spectrum.shape[1]))
    for spec_idx, one_spec in enumerate(spectrum):
        for idx in range(times):
            shuffled_spectrum = np.random.normal(
                loc=one_spec, scale=error[spec_idx])
            shuffled_batch[spec_idx*times+idx, :] = shuffled_spectrum
    return shuffled_batch


def augment_data(spectrum, error, param, times, spectrum_mean, spectrum_std):
    shuffle_x = shuffle_spectrum(spectrum, error, times=times)
    std_x_aug = standardise(shuffle_x, spectrum_mean, spectrum_std)
    std_param_aug = np.repeat(param, times, axis=0)
    return std_x_aug, std_param_aug


def pre_select(gas, ground_truth, org_spectrum, abundance):
    if gas in ['H2O', 'CH4', 'CO', 'CO2', 'NH3']:
        index = (ground_truth > abundance[0]) & (
            ground_truth < abundance[1])
        selected_x = org_spectrum[index]
    elif gas == ['Rp']:
        index = (ground_truth < 1 * R_J)
        selected_x = org_spectrum[index]
    else:
        selected_x = org_spectrum
    return selected_x


def load_history(checkpoint_dir):
    training_log = glob.glob(checkpoint_dir + "history/*.log")
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
