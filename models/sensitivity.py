import numpy as np
import pandas as pd
from .utils import standardise, project_back, random_index
from .ops import pre_select, transform_spectrum
from .constants import R_J, label


def compute_sensitivty_org(model, ground_truth, org_spectrum, org_error, y_data_mean, y_data_std, x_mean, x_std,
                           gases=None, no_spectra=200, repeat=100,  abundance=[-9, -3]):

    # checks

    np.random.seed(42)
    if gases is None:
        gases = label
    spectrum_length = org_spectrum.shape[1]
    # pre-allocate space for squared difference stack
    SD_stack = np.zeros((len(gases), no_spectra, repeat, spectrum_length))
    for idx, gas in enumerate(gases):
        # pre selection, currently doing it element by element
        selected_x = pre_select(
            gas, ground_truth.T[idx], org_spectrum, abundance)
        # main loop
        for i in range(no_spectra):
            # produce ref prediction for a particular spectrum
            org_x = selected_x[i]
            error = org_error[i]
            std_x = standardise(org_x, x_mean, x_std)
            ref_y = model.predict(std_x.reshape(-1, spectrum_length, 1))
            ref_y_org = project_back(ref_y, y_data_mean, y_data_std)[0]
            ref_value = ref_y_org[idx]
            # shuffle given spectrum at different bins and scales.
            for k in range(repeat):
                random_idx = random_index(org_x)
                curr = 0
                selected_bins = spectrum_length
                # current implementation will go from x/2 -> x/4 -> x/8 ... until length <=2
                while selected_bins >= 2:
                    shuffle_x = org_x.copy()
                    new_selected = int(np.ceil(selected_bins/2))
                    picked_index = random_idx[curr:curr+new_selected]
                    # assume uncertainty is Gaussian distributed
                    shift_pt = np.random.normal(
                        loc=shuffle_x[picked_index], scale=error[picked_index])
                    shuffle_x[picked_index] = shift_pt
                    shuffle_x_std = standardise(shuffle_x, x_mean, x_std)
                    y_hat = model.predict(
                        shuffle_x_std.reshape(-1, spectrum_length, 1))
                    y_hat_org = project_back(
                        y_hat, y_data_mean, y_data_std)[0]

                    SD_stack[idx, i, k, picked_index] = y_hat_org[idx]
                    # update index
                    curr += new_selected
                    selected_bins = new_selected

            SD_stack[idx, i, :, :] = np.square(
                SD_stack[idx, i, :, :] - ref_value)

    mean_MSE = np.mean(SD_stack, axis=(1, 2))
    return mean_MSE


def compute_sensitivty_std(model, ground_truth, org_spectrum, org_error, transform,
                           gases=None, no_spectra=200, repeat=100,  abundance=[-4, -3], **kwargs):

    # checks
    if transform == 'standardise':
        trasform = standardise
        arg1 = kwargs['x_mean']
        arg2 = kwargs['x_std']
        arg3 = None
    elif transform == 'transform':
        transform = transform_spectrum
        arg1 = kwargs['baseline_max']
        arg2 = kwargs['baseline_min']
        arg3 = kwargs['qt']

    np.random.seed(42)
    if gases is None:
        gases = label
    spectrum_length = org_spectrum.shape[1]
    # pre-allocate space for squared difference stack
    SD_stack = np.zeros((len(gases), no_spectra, repeat, spectrum_length))
    for idx, gas in enumerate(gases):
        # pre selection, currently doing it element by element
        selected_x = pre_select(
            gas, ground_truth.T[idx], org_spectrum, abundance)
        # main loop
        for i in range(no_spectra):
            # produce ref prediction for a particular spectrum
            org_x = selected_x[i]
            error = org_error[i]
            std_x, *_ = transform(org_x, arg1, arg2, arg3)
            ref_y = model.predict(std_x.reshape(-1, spectrum_length, 1))[0]
            ref_value = ref_y[idx]
            # shuffle given spectrum at different bins and uncertainties.
            for k in range(repeat):
                random_idx = random_index(org_x)
                curr_idx = 0
                selected_bins = spectrum_length
                # current implementation will go from x/2 -> x/4 -> x/8 ... until length <=2
                while selected_bins >= 2:
                    shuffle_x = org_x.copy()
                    new_selected = int(np.ceil(selected_bins/2))
                    picked_index = random_idx[curr_idx:curr_idx+new_selected]
                    # assume uncertainty is Gaussian distributed
                    shift_pt = np.random.normal(
                        loc=shuffle_x[picked_index], scale=error[picked_index])
                    shuffle_x[picked_index] = shift_pt
                    shuffle_x_std, *_ = transform(
                        shuffle_x, arg1, arg2, arg3)
                    y_hat = model.predict(
                        shuffle_x_std.reshape(-1, spectrum_length, 1))[0]
                    SD_stack[idx, i, k, picked_index] = y_hat[idx]
                    # update index
                    curr_idx += new_selected
                    selected_bins = new_selected

            SD_stack[idx, i, :, :] = np.square(
                SD_stack[idx, i, :, :] - ref_value)

    mean_MSE = np.mean(SD_stack, axis=(1, 2))
    return mean_MSE
