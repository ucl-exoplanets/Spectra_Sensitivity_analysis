import yaml
import numpy as np
import pandas as pd
from models import model
from models.utils import standardise, get_random_idx
from models.plotting import sensitivity_plot, return_history
from models.sensitivity import compute_sensitivty_org, compute_sensitivty_std
from models.ops import shuffle_spectrum, load_history, compute_MSE, preprocessing, transform_spectrum

with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# load data
spectrum_file = np.load("data/Massive_ariel/low_X/lowX_spectra.npy")
param_file = pd.read_csv("data/Massive_ariel/low_X/lowX_param_file.csv")
shuffle_times = config['training']['shuffleTimes']

# Extract data from raw file.
spectrum, error, wl, trainable_param = preprocessing(spectrum_file, param_file)

# train test split
train_test_idx = get_random_idx(spectrum, portion=0.8)
x_train_set, x_test = spectrum[train_test_idx], spectrum[~train_test_idx]
y_train_set, y_test = trainable_param[train_test_idx], trainable_param[~train_test_idx]
error_train_set, error_test = error[train_test_idx], error[~train_test_idx]

# train valid split
train_valid_idx = get_random_idx(x_train_set, portion=0.8)
x_org_train, x_org_valid = x_train_set[train_valid_idx], x_train_set[~train_valid_idx]
y_train, y_valid = y_train_set[train_valid_idx], y_train_set[~train_valid_idx]
error_train, error_valid = error_train_set[train_valid_idx], error_train_set[~train_valid_idx]

# Shuffle spectrum
aug_x_train = shuffle_spectrum(x_org_train, error_train, times=shuffle_times)
aug_y_train = np.repeat(y_train, shuffle_times, axis=0)
aug_x_valid = shuffle_spectrum(x_org_valid, error_valid, times=shuffle_times)
aug_y_valid = np.repeat(y_valid, shuffle_times, axis=0)


# Transform input(spectrum)
# spectrum_mean = x_train_set.mean()
# spectrum_std = x_train_set.std()
# std_aug_x_train = standardise(aug_x_train, spectrum_mean, spectrum_std)
# std_aug_x_valid = standardise(aug_x_valid, spectrum_mean, spectrum_std)
# std_x_test = standardise(x_test, spectrum_mean, spectrum_std)

# alternatives
t_spectrum, baseline_max, baseline_min, qt = transform_spectrum(x_org_train)
std_aug_x_train, * \
    _ = transform_spectrum(aug_x_train, baseline_max, baseline_min, qt)
std_aug_x_valid, * \
    _ = transform_spectrum(aug_x_valid, baseline_max, baseline_min, qt)
std_x_test, *_ = transform_spectrum(x_test, baseline_max, baseline_min, qt)

# Transform AMPs
param_mean = y_train_set.mean(axis=0, keepdims=True)
param_std = y_train_set.std(axis=0, keepdims=True)
std_aug_y_train = standardise(aug_y_train, param_mean, param_std)
std_aug_y_valid = standardise(aug_y_valid, param_mean, param_std)
std_y_test = standardise(y_test, param_mean, param_std)

checkpoint_dir = 'output/cnn/test1/'

model = model.Network(param_length=trainable_param.shape[1],
                      spectrum_length=spectrum.shape[1],
                      config=config)
# model.train_model(X_train=std_aug_x_train,
#                   y_train=std_aug_y_train,
#                   X_valid=std_aug_x_valid,
#                   y_valid=std_aug_y_valid,
#                   epochs=5,
#                   lr=0.001,
#                   batch_size=128,
#                   checkpoint_dir=checkpoint_dir,
#                   cv_order=0)

# load model and evaluate
demo_model = model.load_model(checkpoint_dir+'ckt/checkpt_0.h5')
training_loss, valid_loss = load_history(checkpoint_dir)
return_history(training_loss, valid_loss, checkpoint_dir)

model.produce_result(std_x_test, std_y_test, param_mean,
                     param_std, checkpoint_dir)

std_y_pred = model.predict_result(std_x_test)
MSE_score = compute_MSE(std_y_test, std_y_pred)
MSE_score.to_csv(checkpoint_dir+"MSE.csv", index=False)

# Sensitivity analysis
# sensitivity_MSE = compute_sensitivty_org(model=demo_model,
#                                          ground_truth=std_y_test,
#                                          org_spectrum=x_test,
#                                          org_error=error_test,
#                                          y_data_mean=param_mean,
#                                          y_data_std=param_std,
#                                          gases=None,
#                                          no_spectra=5,
#                                          repeat=10,
#                                          x_mean=spectrum_mean,
#                                          x_std=spectrum_std,
#                                          abundance=[-7, -3, ])
sensitivity_MSE = compute_sensitivty_std(model=demo_model,
                                         ground_truth=y_test,
                                         org_spectrum=x_test,
                                         org_error=error_test,
                                         transform='transform',
                                         gases=None,
                                         no_spectra=5,
                                         repeat=10,
                                         abundance=[-7, -3, ],
                                         baseline_max=baseline_max,
                                         baseline_min=baseline_min,
                                         qt=qt)

sensitivity_plot(spectrum=x_test[0],
                 wl=wl,
                 mean_MSE=sensitivity_MSE,
                 checkpoint_dir=checkpoint_dir,
                 fname='sensi_map')
