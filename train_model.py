import yaml
import numpy as np
import pandas as pd
import h5py
import os
from sensitivity_test import model
from sensitivity_test.utils import standardise, get_random_idx
from sensitivity_test.plotting import sensitivity_plot, return_history
from sensitivity_test.sensitivity import compute_sensitivty_org, compute_sensitivty_std
from sensitivity_test.ops import shuffle_spectrum, load_history, compute_MSE, preprocessing, transform_spectrum


def run_DNN(config_path, epochs, lr, batch_size, size):

    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # load data
    spectrum_file = np.load(config['general']['x_data_path'])
    param_file = pd.read_csv(config['general']['y_data_path'])

    # set up variables
    shuffle_times = config['training']['shuffleTimes']
    checkpoint_dir = config['general']['checkpoint_dir']
    seed = config['general']['seed']
    os.makedirs(checkpoint_dir, exist_ok=True)

    # initialise save file
    f = h5py.File(checkpoint_dir+"result_file.hdf5", "w")

    # preprocessing
    spectrum, error, wl, trainable_param, Rstar = preprocessing(
        spectrum_file, param_file, Rs=False, size=size)

    # train test split
    train_test_idx = get_random_idx(spectrum, portion=0.8, seed=seed)
    x_train_set, x_test = spectrum[train_test_idx], spectrum[~train_test_idx]
    y_train_set, y_test = trainable_param[train_test_idx], trainable_param[~train_test_idx]
    error_train_set, error_test = error[train_test_idx], error[~train_test_idx]
    # Rster_train_set, Rster_test = Rstar[train_test_idx], Rstar[~train_test_idx]

    for cv in range(config['training']['cv']):
        seed += cv
        # train valid split
        train_valid_idx = get_random_idx(x_train_set, portion=0.8, seed=seed)
        x_org_train, x_org_valid = x_train_set[train_valid_idx], x_train_set[~train_valid_idx]
        y_train, y_valid = y_train_set[train_valid_idx], y_train_set[~train_valid_idx]
        error_train, error_valid = error_train_set[train_valid_idx], error_train_set[~train_valid_idx]
        # Rster_train, Rster_valid = Rster_train_set[train_valid_idx], Rster_train_set[~train_valid_idx]

        # Shuffle spectrum
        aug_x_train = shuffle_spectrum(
            x_org_train, error_train, times=shuffle_times)
        aug_y_train = np.repeat(y_train, shuffle_times, axis=0)
        aug_x_valid = shuffle_spectrum(
            x_org_valid, error_valid, times=shuffle_times)
        aug_y_valid = np.repeat(y_valid, shuffle_times, axis=0)
        # aug_Rstar_train = np.repeat(Rster_train, shuffle_times, axis=0)
        # aug_Rstar_valid = np.repeat(Rster_valid, shuffle_times, axis=0)

        # Transform input (spectrum)
        spectrum_mean = x_train_set.mean()
        spectrum_std = x_train_set.std()
        std_aug_x_train = standardise(aug_x_train, spectrum_mean, spectrum_std)
        std_aug_x_valid = standardise(aug_x_valid, spectrum_mean, spectrum_std)
        std_x_test = standardise(x_test, spectrum_mean, spectrum_std)
        std_x_valid = standardise(x_org_valid, spectrum_mean, spectrum_std)

        # Transform input (AMPs)
        param_mean = y_train_set.mean(axis=0, keepdims=True)
        param_std = y_train_set.std(axis=0, keepdims=True)
        std_aug_y_train = standardise(aug_y_train, param_mean, param_std)
        std_aug_y_valid = standardise(aug_y_valid, param_mean, param_std)
        std_y_test = standardise(y_test, param_mean, param_std)
        std_y_valid = standardise(y_valid, param_mean, param_std)

        # Transform Rstar (only used if extra input = True)
        # Rs_mean = Rster_train_set.mean(axis=0, keepdims=True)
        # Rs_std = Rster_train_set.std(axis=0, keepdims=True)
        # std_aug_Rs_train = standardise(aug_Rstar_train, Rs_mean, Rs_std)
        # std_aug_Rs_valid = standardise(aug_Rstar_valid, Rs_mean, Rs_std)
        # std_Rs_test = standardise(Rster_test, Rs_mean, Rs_std)

        DNN = model.Network(param_length=trainable_param.shape[1],
                            spectrum_length=spectrum.shape[1],
                            config=config)
        if config['training']['train']:
            DNN.train_model(X_train=std_aug_x_train,
                            y_train=std_aug_y_train,
                            X_valid=std_aug_x_valid,
                            y_valid=std_aug_y_valid,
                            epochs=epochs,
                            lr=lr,
                            batch_size=batch_size,
                            checkpoint_dir=checkpoint_dir,
                            cv_order=cv)
        # else:
        #     DNN.load_model(checkpoint_dir+"ckt/checkpt_0.h5")
        current_model = DNN.load_model(checkpoint_dir+f'ckt/checkpt_{cv}.h5')
        std_y_pred = DNN.predict_result(std_x_test, std_y_test)
        if cv == 0:
            master_mse = compute_MSE(std_y_test, std_y_pred)
            org_y_test = f.create_dataset("org_y_test", data=y_test)
        else:
            MSE_score = compute_MSE(std_y_test, std_y_pred)
            master_mse = master_mse.append(
                MSE_score.iloc[0], ignore_index=True)

        # save results
        trial = f.create_group(f"trial_{cv}")
        y_pred = trial.create_dataset(f"std_y_pred_{cv}", data=std_y_pred)
        mean = trial.create_dataset(f"param_mean_{cv}", data=param_mean)
        std = trial.create_dataset(f"param_std_{cv}", data=param_std)

        # Produce various model diagnostics
        if config['general']['run_diagnostics']:
            DNN.produce_result(std_x_test, std_y_test, param_mean,
                               param_std, checkpoint_dir, order=cv)
        # Sensitivity analysis
        if config['general']['run_sensitivity']:
            sensitivity_MSE = compute_sensitivty_org(model=current_model,
                                                     ground_truth=y_test,
                                                     org_spectrum=x_test,
                                                     org_error=error_test,
                                                     y_data_mean=param_mean,
                                                     y_data_std=param_std,
                                                     gases=None,
                                                     no_spectra=300,
                                                     repeat=1000,
                                                     x_mean=spectrum_mean,
                                                     x_std=spectrum_std,
                                                     abundance=[-7, -3, ])

            sensitivity_plot(spectrum=x_test[0],
                             wl=wl,
                             mean_MSE=sensitivity_MSE,
                             checkpoint_dir=checkpoint_dir,
                             fname='sensi_map')
            np.save(checkpoint_dir+f"results/sensi_map_{cv}", sensitivity_MSE)

    master_mse = master_mse.append(master_mse.mean(axis=0), ignore_index=True)
    master_mse = master_mse.rename(index={cv+1: 'Average'})
    master_mse.to_csv(checkpoint_dir+"MSE.csv", index=True)

    # Plot progress bar
    training_loss, valid_loss = load_history(checkpoint_dir)
    return_history(training_loss, valid_loss, checkpoint_dir)
