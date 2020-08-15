from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import glob
from .constants import M_J, R_J, label, units


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


def plot_sensitivity(wl, spectrum, mean_std, checkpoint_dir, order=0,
                     gases=None, name='sensi_map'):
    gas_contri = np.load("./data/all_gas_contri.npy")
    gas_id = {'H2O': 0, 'CH4': 1, 'CO': 2, 'CO2': 3, 'NH3': 4}
    if gases is None:
        gases = label
    for idx, gas in enumerate(gases):
        plt.figure(figsize=(10, 8))
        if gas in ['H2O', 'CH4', 'CO', 'CO2', 'NH3']:
            ax1 = plt.subplot(2, 1, 1)
        else:
            ax1 = plt.subplot(1, 1, 1)
        if gas == 'Rp':
            mean_std[idx] = mean_std[idx]/R_J
        q = np.quantile(mean_std[idx], 0.9)

        plt.scatter(wl, spectrum, c=mean_std[idx], vmax=q)
        plt.plot(wl, spectrum)
        plt.xscale('log')
        plt.xticks([1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7])
        plt.xlim([wl.min()*0.9, wl.max()*1.05])
        plt.ylim([spectrum.min()*0.95, spectrum.max()*1.05])
        plt.colorbar(orientation='horizontal')
        plt.title(
            f'Average {name} for {gas} (Spectrum view)')
        if gas in ['H2O', 'CH4', 'CO', 'CO2', 'NH3']:
            ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            plt.scatter(
                wl, gas_contri[gas_id[gas]], marker='o', c=mean_std[idx], vmax=q)
            plt.plot(wl, gas_contri[gas_id[gas]])
            plt.xscale('log')
            plt.ylim([gas_contri[gas_id[gas]].min()*0.95,
                      gas_contri[gas_id[gas]].max()*1.05])
            plt.xlim([wl.min() * 0.9, wl.max() * 1.05])
            plt.xticks([1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7])
            plt.title(f'Contribution Function for {gas}')

        plt.savefig(os.path.join(
            checkpoint_dir, f'results/{name}_{gas}_{order}.png'))
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


def sensitivity_plot(mean_MSE, wl, spectrum, checkpoint_dir, fname="map"):

    fig = plt.figure(figsize=(17, 13))
    gas_contri = np.load("./data/all_gas_contri.npy")

    plt.subplot(1, 2, 1)
    color = ['black', 'black', 'black', 'black', 'black']
    name = ['H$_2$O', 'CH$_4$', 'CO', 'CO$_2$', 'NH$_3$']
    offset = [0, 0.001, 0.0015, 0.0015, 0.0015]
    t_offset = [0, 0.001, 0.0015, 0.0015, 0.0015]
    for i in range(5):
        plt.text(x=0.6, y=0.0049+t_offset[i]*i,
                 s=name[i], color=color[i], fontsize=16)
        q = np.quantile(mean_MSE[i], 0.95)
        plt.plot(wl, gas_contri[i]+(offset[i]*i),
                 c=color[i], label=name[i], alpha=0.5, zorder=1)
        ax = plt.scatter(wl, gas_contri[i]+(offset[i]*i),
                         marker='o', c=mean_MSE[i], vmax=q, zorder=99)
        plt.xscale('log')
        plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], [
                   1, 2, 3, 4, 5, 6, 7, 8], fontsize=12)
        plt.xlabel('Wavelength ($\mu m$)', fontsize=16)
        plt.yticks([])

    plt.subplot(1, 2, 2)
    color = ['black', 'black', 'black', 'black']
    name = ['M$_p$', 'R$_p$', 'T$_p$', 'Cloud']
    offset = 0.002
    for idx, i in enumerate(range(5, 9)):
        plt.text(x=0.6, y=0.0037+offset*idx,
                 s=name[idx], color=color[idx], fontsize=16)
        q = np.quantile(mean_MSE[i], 0.95)
        lq = np.quantile(mean_MSE[i], 0.0)
        plt.plot(wl, spectrum+(offset*idx),
                 c=color[idx], label=name[idx], alpha=0.5, zorder=1)
        ax = plt.scatter(wl, spectrum+(offset*idx), marker='o',
                         c=mean_MSE[i], vmax=q, zorder=99)
    plt.xscale('log')
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], fontsize=12)
    plt.xlabel('Wavelength ($\mu m$)', fontsize=16)
    plt.yticks([])
    plt.subplots_adjust(left=0.125,  # the left side of the subplots of the figure
                        right=0.9,
                        bottom=0.1,
                        top=0.87,
                        wspace=0.1,
                        hspace=0.18, )
    cbar_ax = fig.add_axes([0.165, 0.92, 0.7, 0.03])
    cbar = fig.colorbar(ax, cax=cbar_ax, ticks=[
                        lq, q], orientation='horizontal')
    cbar.ax.set_xticklabels(['Least Sensitive', 'Most Sensitive'], fontsize=13)
    plt.savefig(checkpoint_dir + f"{fname}.png")
