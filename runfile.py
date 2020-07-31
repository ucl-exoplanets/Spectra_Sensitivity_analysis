import yaml
from model import Network
import numpy as np
import pandas as pd
import utils
import plotting
import ops

with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# load data
spectrum_file = np.load("data/Massive_ariel/low_X/lowX_spectra.npy")
param_file = pd.read_csv("data/Massive_ariel/low_X/lowX_param_file.csv")

# Pre-process data
trainable_param = param_file.values[:, :9]
spectrum = spectrum_file[:, 0, :]
error = spectrum_file[:, 1, :]
wl = spectrum_file[0, 2, :]

# Molecules, Mp and clouds must be in log-scale
trainable_param[:, :6] = np.log10(trainable_param[:, :6])
trainable_param[:, -1] = np.log10(trainable_param[:, -1])

# train test split
train_test_idx = utils.get_random_idx(spectrum, portion=0.8)
x_train_set, y_train_set = spectrum_file[train_test_idx], trainable_param[train_test_idx]
x_test_set, y_test = spectrum_file[~train_test_idx], trainable_param[~train_test_idx]
x_test = spectrum[~train_test_idx]

# Extract training global mean and std values
spectrum_mean = x_train_set[:, 0, :].mean()
spectrum_std = x_train_set[:, 0, :].std()
std_x_test = utils.standardise(x_test, spectrum_mean, spectrum_std)

# standardise AMPs
param_mean = y_train_set.mean(axis=0, keepdims=True)
param_std = y_train_set.std(axis=0, keepdims=True)
std_y_train_set = utils.standardise(y_train_set, param_mean, param_std)
std_y_test = utils.standardise(y_test, param_mean, param_std)

# train valid split
train_valid_idx = utils.get_random_idx(x_train_set, portion=0.8)
x_org_train, y_train = x_train_set[train_valid_idx], std_y_train_set[train_valid_idx]
x_org_valid, y_valid = x_train_set[~train_valid_idx], std_y_train_set[~train_valid_idx]

# augment train and validation data
std_x_aug_train, std_y_aug_train = utils.augment_data(spectrum=x_org_train[:, 0, :],
                                                      error=x_org_train[:, 1, :],
                                                      param=y_train,
                                                      times=config['training']['shuffleTimes'],
                                                      spectrum_mean=spectrum_mean, spectrum_std=spectrum_std)

std_x_aug_valid, std_y_aug_valid = utils.augment_data(spectrum=x_org_valid[:, 0, :],
                                                      error=x_org_valid[:, 1, :],
                                                      param=y_valid,
                                                      times=config['training']['shuffleTimes'],
                                                      spectrum_mean=spectrum_mean, spectrum_std=spectrum_std)

checkpoint_dir = 'output/cnn/test1/'
model = Network(
    param_length=trainable_param.shape[1], spectrum_length=spectrum.shape[1], config=config)
# model.train_model(X_train=std_x_aug_train,
#                   y_train=std_y_aug_train,
#                   X_valid=std_x_aug_valid,
#                   y_valid=std_y_aug_valid,
#                   epochs=5,
#                   lr=0.001,
#                   batch_size=128,
#                   checkpoint_dir=checkpoint_dir,
#                   cv_order=0)

# todo after training analysis.
demo_model = model.load_model(checkpoint_dir+'ckt/checkpt_0.h5')
training_loss, valid_loss = utils.load_history(checkpoint_dir)
plotting.return_history(training_loss, valid_loss, checkpoint_dir)

model.produce_result(std_x_test, std_y_test, param_mean,
                     param_std, checkpoint_dir)

std_y_pred = model.predict_result(std_x_test)
MSE_score = utils.compute_MSE(std_y_test, std_y_pred)
MSE_score.to_csv(checkpoint_dir+"MSE.csv", index=False)

# Sensitivity analysis
sensitivity_MSE = ops.compute_sensitivty_org(model=demo_model,
                                             y_test=std_y_test,
                                             org_spectrum=x_test,
                                             org_error=x_test_set[:, 1, :],
                                             y_data_mean=param_mean,
                                             y_data_std=param_std,
                                             gases=None,
                                             no_spectra=5,
                                             repeat=10,
                                             x_mean=spectrum_mean,
                                             x_std=spectrum_std,
                                             abundance=[-7, -3, ])

plotting.sensitivity_plot(spectrum=x_test[0],
                          wl=wl,
                          mean_MSE=sensitivity_MSE,
                          checkpoint_dir=checkpoint_dir,
                          name='sensi_map')
