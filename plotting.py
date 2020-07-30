from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import glob
from models.ops import *
from constants import M_J, R_J, label, units


def plot_compare_truth(y_test_org, y_predict_org, checkpoint_dir, order=0,
                       scale=None, chosen_gas=None, alpha=0.4):

    assert y_test_org.shape[1] == y_predict_org.shape[1]

    elements = y_test_org.shape[1]

    if chosen_gas is None:
        gas = label
    else:
        gas = chosen_gas

    if len(gas) == 1:
        fig = plt.figure(figsize=(8, 8))
    else:
        fig = plt.figure(figsize=(18, 15))

    fig.suptitle(
        "True and predicted abundance in different quantities", fontsize=16)

    if elements > 3:
        row, col = 3, int(np.ceil(len(gas) / 3))
    else:
        row, col = len(gas), 1

    for i in range(y_test_org.shape[1]):
        ax = plt.subplot(row, col, i + 1)
        sorted_id = np.argsort(y_test_org.T[i])
        sort_ydata = y_test_org.T[i][sorted_id]
        sort_ypredict = y_predict_org.T[i][sorted_id]

        if scale is not None:
            scale = scale[sorted_id]
        else:
            scale = abs(sort_ydata - sort_ypredict)

        if gas[i] in ['H2O', 'CH4', 'CO', 'CO2', 'NH3']:
            plt.scatter(x=sort_ydata, y=sort_ypredict, c=scale, marker='v', label="Predicted Abundance",
                        s=20, alpha=alpha, zorder=1)
            plt.plot([sort_ydata.min(), sort_ydata.max()], [sort_ydata.min(), sort_ydata.max()], zorder=99, lw=3,
                     c='black')

            plt.ylim([sort_ydata.min() * 1.01, sort_ydata.max() * 0.95])
            plt.xlim([sort_ydata.min() * 1.01, sort_ydata.max() * 0.95])
        elif gas[i] == 'Mp':
            ydata_jpmass = (10 ** sort_ydata) / M_J
            ypredict_jpmass = (10 ** sort_ypredict) / M_J
            plt.scatter(x=ydata_jpmass, y=ypredict_jpmass, c=scale, marker='v',
                        label="Predicted Abundance", s=20, alpha=alpha, zorder=1)
            plt.plot([ydata_jpmass.min(), ydata_jpmass.max()], [ydata_jpmass.min(), ydata_jpmass.max()], zorder=99,
                     lw=3, c='black')

            plt.ylim([ydata_jpmass.min() * 0.95, ydata_jpmass.max() * 1.01])
            plt.xlim([ydata_jpmass.min() * 0.95, ydata_jpmass.max() * 1.01])
            plt.xscale('log')
            plt.yscale('log')

        elif gas[i] == 'Rp':
            ydata_in_Jr = sort_ydata / R_J
            ypredict_in_Jr = sort_ypredict / R_J
            plt.scatter(x=ydata_in_Jr, y=ypredict_in_Jr, c=scale, marker='v', label="Predicted Abundance",
                        s=20, alpha=alpha, zorder=1)
            plt.plot([ydata_in_Jr.min(), ydata_in_Jr.max()], [ydata_in_Jr.min(), ydata_in_Jr.max()], zorder=99,
                     lw=3, c='black')
            plt.ylim([ydata_in_Jr.min() * 0.95, ydata_in_Jr.max() * 1.01])
            plt.xlim([ydata_in_Jr.min() * 0.95, ydata_in_Jr.max() * 1.01])

        elif gas[i] == 'Tp':
            plt.scatter(x=sort_ydata, y=sort_ypredict, c=scale, marker='v', label="Predicted Abundance", s=20,
                        alpha=alpha, zorder=1)
            plt.plot([sort_ydata.min(), sort_ydata.max()], [sort_ydata.min(), sort_ydata.max()], zorder=99, lw=3,
                     c='black')
            plt.ylim([sort_ydata.min() * 0.8, sort_ydata.max() * 1.01])
            plt.xlim([sort_ydata.min() * 0.8, sort_ydata.max() * 1.01])

        elif gas[i] == 'Cloud Top Pressure':
            plt.scatter(x=sort_ydata, y=sort_ypredict, c=scale, marker='v', label="Predicted Abundance", s=20,
                        alpha=alpha, zorder=1)
            plt.plot([sort_ydata.min(), sort_ydata.max()], [sort_ydata.min(), sort_ydata.max()], zorder=99, lw=3,
                     c='black')
            plt.ylim([sort_ydata.min() * 0.8, sort_ydata.max() * 1.01])
            plt.xlim([sort_ydata.min() * 0.8, sort_ydata.max() * 1.01])

        plt.colorbar()
        plt.title(f"{gas[i]}")
        plt.ylabel(f'Predicted {units[i]}')
        plt.xlabel(f'True {units[i]}')
    plt.savefig(os.path.join(checkpoint_dir, f"results/compare_truth_{order}"))
    plt.close()


def plot_sensitivity(full_spectrum, mean_std, checkpoint_dir, order=0,
                     chosen_gas=None, name='sensi_map', abundance=[-4, -3]):
    gas_contri = np.load("./data/all_gas_contri.npy")
    Jup_radius = 69911000
    gas_id = {'H2O': 0, 'CH4': 1, 'CO': 2, 'CO2': 3, 'NH3': 4}
    for idx, gas in enumerate(chosen_gas):
        plt.figure(figsize=(10, 8))
        if gas in ['H2O', 'CH4', 'CO', 'CO2', 'NH3']:
            ax1 = plt.subplot(2, 1, 1)
        else:
            ax1 = plt.subplot(1, 1, 1)
        if gas == 'Rp':
            mean_std[idx] = mean_std[idx]/Jup_radius
        q = np.quantile(mean_std[idx], 0.9)

        plt.scatter(full_spectrum[0, 2, :],
                    full_spectrum[0, 0, :], c=mean_std[idx], vmax=q)
        plt.plot(full_spectrum[0, 2, :], full_spectrum[0, 0, :])
        plt.xscale('log')
        plt.xticks([1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7])
        plt.xlim([full_spectrum[0, 2, :].min()*0.9,
                  full_spectrum[0, 2, :].max()*1.05])
        plt.ylim([full_spectrum[0, 0, :].min()*0.95,
                  full_spectrum[0, 0, :].max()*1.05])
        plt.colorbar(orientation='horizontal')
        plt.title(
            f'Average {name} for {gas} (logX = [{abundance[0]},{abundance[1]}]) (Spectrum view)')
        if gas in ['H2O', 'CH4', 'CO', 'CO2', 'NH3']:
            ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            plt.scatter(
                full_spectrum[0, 2, :], gas_contri[gas_id[gas]], marker='o', c=mean_std[idx], vmax=q)
            plt.plot(full_spectrum[0, 2, :], gas_contri[gas_id[gas]])
            plt.xscale('log')
            plt.ylim([gas_contri[gas_id[gas]].min()*0.95,
                      gas_contri[gas_id[gas]].max()*1.05])
            plt.xlim([full_spectrum[0, 2, :].min() * 0.9,
                      full_spectrum[0, 2, :].max() * 1.05])
            plt.xticks([1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7])
            plt.title(f'Contribution Function for {gas}')

        if not os.path.exists(os.path.join(checkpoint_dir, f'results/logX_{abundance[0]}_{abundance[1]}')):
            os.makedirs(os.path.join(checkpoint_dir,
                                     f'results/logX_{abundance[0]}_{abundance[1]}'))
        plt.savefig(os.path.join(
            checkpoint_dir, f'results/logX_{abundance[0]}_{abundance[1]}/{name}_{gas}_{order}.png'))
        plt.close()


def plot_bias_varience(full_spectrum, total_std, total_mean, checkpoint_dir, order=0,
                       chosen_gas=None, name='bias_var_', abundance=[-4, -3]):
    gas_contri = np.load("./data/all_gas_contri.npy")
    gas_id = {'H2O': 0, 'CH4': 1, 'CO': 2, 'CO2': 3, 'NH3': 4}
    Jup_radius = 69911000
    for idx, gas in enumerate(chosen_gas):
        total_mean = np.abs(total_mean)
        total_std = np.abs(total_std)
        if gas == 'Rp':
            total_mean[idx] = total_mean[idx]/Jup_radius
            total_std[idx] = total_std[idx] / Jup_radius
        q_mean = np.quantile(total_mean[idx], 0.9)
        q_std = np.quantile(total_std[idx], 0.9)
        plt.figure(figsize=(16, 8))
        if gas in ['H2O', 'CH4', 'CO', 'CO2', 'NH3']:
            ax1 = plt.subplot(2, 2, 1)
        else:
            ax1 = plt.subplot(2, 1, 1)

        plt.scatter(full_spectrum[0, 2, :], full_spectrum[0,
                                                          0, :], c=total_mean[idx], vmax=q_mean)
        plt.plot(full_spectrum[0, 2, :], full_spectrum[0, 0, :])
        plt.xscale('log')
        plt.xlim([full_spectrum[0, 2, :].min() * 0.9,
                  full_spectrum[0, 2, :].max() * 1.05])
        plt.ylim([full_spectrum[0, 0, :].min() * 0.95,
                  full_spectrum[0, 0, :].max() * 1.05])
        plt.title(
            f'MAE (top) and Variance (bottom) for {gas} (logX = [{abundance[0]},{abundance[1]}])(Spectrum View)')
        plt.yticks([], [])
        if gas in ['H2O', 'CH4', 'CO', 'CO2', 'NH3']:
            pass
        else:
            plt.colorbar(orientation='vertical', shrink=0.9, pad=0.02)

        if gas in ['H2O', 'CH4', 'CO', 'CO2', 'NH3']:
            ax3 = plt.subplot(2, 2, 3, sharex=ax1)
        else:
            ax3 = plt.subplot(2, 1, 2, sharex=ax1)

        plt.scatter(
            full_spectrum[0, 2, :], full_spectrum[0, 0, :], c=total_std[idx], vmax=q_std)
        plt.plot(full_spectrum[0, 2, :], full_spectrum[0, 0, :])
        plt.xscale('log')
        plt.xticks([1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7])
        plt.xlim([full_spectrum[0, 2, :].min() * 0.9,
                  full_spectrum[0, 2, :].max() * 1.05])
        plt.ylim([full_spectrum[0, 0, :].min() * 0.95,
                  full_spectrum[0, 0, :].max() * 1.05])
        plt.yticks([], [])

        if gas in ['H2O', 'CH4', 'CO', 'CO2', 'NH3']:
            pass
        else:
            plt.colorbar(orientation='vertical', shrink=0.9, pad=0.02)

        if gas in ['H2O', 'CH4', 'CO', 'CO2', 'NH3']:
            ax2 = plt.subplot(2, 2, 2, sharex=ax1)
            plt.colorbar(orientation='vertical', shrink=0.9, pad=0.02)
            plt.scatter(full_spectrum[0, 2, :], gas_contri[gas_id[gas]],
                        marker='o', c=total_mean[idx], vmax=q_mean)
            plt.plot(full_spectrum[0, 2, :], gas_contri[gas_id[gas]])
            plt.xscale('log')
            plt.ylim([gas_contri[gas_id[gas]].min() * 0.95,
                      gas_contri[gas_id[gas]].max() * 1.05])
            plt.xlim([full_spectrum[0, 2, :].min() * 0.9,
                      full_spectrum[0, 2, :].max() * 1.05])
            plt.title(
                f'Bias (top) and Variance (bottom) for {gas} (Contribution View)')
            plt.yticks([], [])

            ax4 = plt.subplot(2, 2, 4, sharex=ax1)
            plt.colorbar(orientation='vertical', shrink=0.9, pad=0.02)
            plt.scatter(full_spectrum[0, 2, :], gas_contri[gas_id[gas]],
                        marker='o', c=total_std[idx], vmax=q_std)
            plt.plot(full_spectrum[0, 2, :], gas_contri[gas_id[gas]])
            plt.xscale('log')
            plt.ylim([gas_contri[gas_id[gas]].min() * 0.95,
                      gas_contri[gas_id[gas]].max() * 1.05])
            plt.xlim([full_spectrum[0, 2, :].min() * 0.9,
                      full_spectrum[0, 2, :].max() * 1.05])
            plt.xticks([1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7])
            plt.yticks([], [])
        if not os.path.exists(os.path.join(checkpoint_dir, f'results/logX_{abundance[0]}_{abundance[1]}')):
            os.makedirs(os.path.join(checkpoint_dir,
                                     f'results/logX_{abundance[0]}_{abundance[1]}'))
        plt.savefig(os.path.join(
            checkpoint_dir, f'results/logX_{abundance[0]}_{abundance[1]}/{name}_{gas}_{order}.png'))
        plt.subplots_adjust(wspace=0, hspace=0.02)

        plt.close()


def return_history(training_loss, valid_loss, checkpoint_dir):

    std_loss = np.std(training_loss, axis=0)
    std_valid_loss = np.std(valid_loss, axis=0)
    mean_loss = np.mean(training_loss, axis=0)
    mean_valid_loss = np.mean(valid_loss, axis=0)
    epoch = np.arange(len(mean_loss))

    plt.figure(figsize=(8, 6))
    plt.errorbar(x=epoch, y=mean_loss,
                 yerr=std_loss, label="training_loss")
    plt.errorbar(x=epoch, y=mean_valid_loss,
                 yerr=std_valid_loss, label="val_loss")
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, "progress_plot"))
    plt.close()
