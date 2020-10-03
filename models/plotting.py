from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pandas as pd
import glob
from .constants import M_J, R_J, label, units
from .ops import get_equal_bin


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

        if gas[i] in ['H$_2$O', 'CH$_4$', 'CO', 'CO$_2$', 'NH$_3$']:
            plt.scatter(x=sort_ydata, y=sort_ypredict, c=scale, marker='v', label="Predicted Abundance",
                        s=20, alpha=alpha, zorder=1)
            plt.plot([sort_ydata.min(), sort_ydata.max()], [sort_ydata.min(), sort_ydata.max()], zorder=99, lw=3,
                     c='black')

            plt.ylim([sort_ydata.min() * 1.01, sort_ydata.max() * 0.95])
            plt.xlim([sort_ydata.min() * 1.01, sort_ydata.max() * 0.95])
        elif gas[i] == 'M$_p$':
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

        elif gas[i] == "R$_p$":
            ydata_in_Jr = sort_ydata / R_J
            ypredict_in_Jr = sort_ypredict / R_J
            plt.scatter(x=ydata_in_Jr, y=ypredict_in_Jr, c=scale, marker='v', label="Predicted Abundance",
                        s=20, alpha=alpha, zorder=1)
            plt.plot([ydata_in_Jr.min(), ydata_in_Jr.max()], [ydata_in_Jr.min(), ydata_in_Jr.max()], zorder=99,
                     lw=3, c='black')
            plt.ylim([ydata_in_Jr.min() * 0.95, ydata_in_Jr.max() * 1.01])
            plt.xlim([ydata_in_Jr.min() * 0.95, ydata_in_Jr.max() * 1.01])

        elif gas[i] == "T$_p$":
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
    plt.savefig(os.path.join(checkpoint_dir,
                             f"results/compare_truth_{order}.pdf"))
    plt.close()


def BVPlot(y_test_org, y_predict_org, checkpoint_dir, order=0, chosen_gas=None, color='#1E90FF', batch_size=100):
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20, 16))
    ax = ax.flatten()
    if chosen_gas is None:
        gas = label
    else:
        gas = chosen_gas

    for mol in range(y_test_org.shape[1]):
        binned_x, binned_y, binned_yerr, _ = get_equal_bin(
            y_test_org[:, mol], y_predict_org[:, mol], batch_size)

        if gas[mol] == 'R$_p$':
            ax[mol].errorbar(x=binned_x/R_J, y=binned_y/R_J,
                             marker='o', yerr=binned_yerr/R_J, ls='-', color=color)

            ax[mol].set_ylabel('Average Deviation [R$_J$]', fontsize=15)
        elif gas[mol] == 'M$_p$':
            binned_x, binned_y, binned_yerr, _ = get_equal_bin(
                10**y_test_org[:, mol], 10**y_predict_org[:, mol], batch_size)

            ax[mol].errorbar(x=binned_x/M_J, y=binned_y/M_J,
                             marker='o', yerr=binned_yerr/M_J, ls='-', color=color)

            ax[mol].set_xscale('log')
            ax[mol].set_ylabel('Average Deviation [M$_J$]', fontsize=15)
            ax[mol].set_ylim(ymin=0, ymax=3.5)
            ax[mol].set_xticks([0.1, 1, 2, 3, 4])
            ax[mol].get_xaxis().set_major_formatter(ticker.ScalarFormatter())

        elif gas[mol] == "T$_p$":
            ax[mol].errorbar(x=binned_x, y=binned_y, marker='o',
                             yerr=binned_yerr, color=color, zorder=99, alpha=0.7)
            ax[mol].set_ylabel('Average Deviation [K]', fontsize=15)

        else:
            ax[mol].errorbar(x=binned_x, y=binned_y, marker='o',
                             yerr=binned_yerr, color=color, zorder=99, alpha=0.7)
            ax[mol].set_ylabel('Average Deviation', fontsize=15)

        ax[mol].set_xlabel(f"True {units[mol]}", fontsize=15)
        ax[mol].set_title(f"{gas[mol]}", fontsize=20)

    plt.subplots_adjust(left=0.125,
                        right=0.9,
                        bottom=0.1,
                        top=0.9,
                        wspace=0.35,
                        hspace=0.25)

    plt.tight_layout()
    fig.savefig(os.path.join(checkpoint_dir, f"results/BVPlot_{order}.pdf"))
    plt.close()


def pred_deviation_plot(y_test_org, y_predict_org, checkpoint_dir, order=0, chosen_gas=None, color='#6495ED'):
    fig, axs = plt.subplots(nrows=3, ncols=3, sharex=False,
                            sharey=False, figsize=(20, 14), squeeze=True)
    axs = axs.flatten()
    if chosen_gas is None:
        gas = label
    else:
        gas = chosen_gas
    for i in range(y_test_org.shape[1]):
        y_pred = y_predict_org[:, i]
        y_actual = y_test_org[:, i]

        if gas[i] == 'M$_p$':
            y_pred = y_pred - np.log10(M_J)
            y_actual = y_actual - np.log10(M_J)
            step = -0.3
            width = 0.1
        elif gas[i] == 'R$_p$':
            y_pred = y_pred / R_J
            y_actual = y_actual / R_J
            step = -0.2
            width = 0.03
        elif gas[i] == 'T$_p$':
            step = -300
            width = 50
        elif gas[i] == 'Cloud Top Pressure':
            step = -0.3
            width = 0.12
        else:
            step = -0.5
            width = 0.5

        abn_list = np.arange(y_actual.max(), y_actual.min(), step)

        devi_list = np.zeros((len(abn_list)))
        std_list = np.zeros((len(abn_list)))
        bin_width = []
        counts = np.zeros((len(abn_list)))
        # TODO put into a function
        for idx, abn in enumerate(abn_list):
            selected = (y_pred < abn) & (y_pred > abn+step)
            if selected.sum() < 20:
                devi_list[idx] = np.nan
                std_list[idx] = np.nan
                bin_width.append(step)
                continue
            pred = y_pred[selected]
            actual = y_actual[selected]
            counts[idx] = len(pred)
            L1 = np.abs(pred - actual)
            L1_mean = np.mean(L1)
            L1_std = 1.96*np.std(L1)/np.sqrt(len(pred))
            devi_list[idx] = L1_mean
            std_list[idx] = L1_std
            bin_width.append(step)

        x_bins = abn_list+step/2
        devi_list[np.isnan(devi_list)] = 0

        axs[i].bar(x_bins, devi_list, width=bin_width,
                   yerr=std_list, color=color, edgecolor='black')
        axs[i].set_title(f"{label[i]}", fontsize=18)
        ax2 = axs[i].twinx()
        ax2.plot(x_bins, counts, color='black', alpha=0.6, ls='-')
        ax2.scatter(x_bins, counts, color='black', alpha=0.6, marker='x')
        ax2.set_ylim(bottom=0, top=1300)
        ax2.set_ylabel("Frequency", fontsize=14)

        axs[i].set_ylabel("Average Deviation", fontsize=14)
        axs[i].set_xlabel(f"Predicted {units[i]}", fontsize=14)

    plt.subplots_adjust(left=0.125,
                        right=0.9,
                        bottom=0.1,
                        top=0.9,
                        wspace=0.2,
                        hspace=0.35)
    plt.tight_layout()
    fig.savefig(os.path.join(checkpoint_dir, f"results/deviation_{order}.pdf"))
    plt.close()


def credibility_plot(y_test_org, y_predict_org, checkpoint_dir, order=0, eps=0.5, delta=0.7, color='#FF6347'):
    eps = 0.5
    delta = 0.7

    fig, axs = plt.subplots(nrows=5, ncols=1, sharex=False,
                            sharey=True, figsize=(8, 12), squeeze=True)
    axs = axs.flatten()
    for i in range(5):

        y_pred = y_predict_org[:, i]
        y_actual = y_test_org[:, i]
        step = -0.4
        width = 0.4

        abn_list = np.arange(y_actual.max(), y_actual.min(), step)
        probability_list = np.zeros((len(abn_list)))

        for idx, abn in enumerate(abn_list):
            selected = (y_pred < abn) & (y_pred > abn+step)
            if selected.sum() < 20:
                continue
            pred = y_pred[selected]
            actual = y_actual[selected]
            # Credibility calculation
            difference = np.abs(pred - actual)
            items = difference < eps
            total_pred_num = len(pred)
            P = items.sum()/total_pred_num
            probability_list[idx] = P
        x_bins = abn_list+step/2
        axs[i].bar(x_bins, probability_list, width=width,
                   color=color, edgecolor='black')
        axs[i].set_title(f"{label[i]}", fontsize=18)
        axs[i].set_ylim(bottom=0, top=1)
        axs[i].set_ylabel("Probability", fontsize=14)
        if i == 4:
            axs[i].set_xlabel(f"Predicted Abundance level", fontsize=14)
        axs[i].axhline(delta, color="#778899", lw=3, ls='--')

    plt.tight_layout()
    fig.savefig(os.path.join(checkpoint_dir, f"results/cred_plot_{order}.pdf"))
    plt.close()


def plot_sensitivity(wl, spectrum, mean_std, checkpoint_dir, order=0,
                     gases=None, name='sensi_map'):
    gas_contri = np.load("./data/all_gas_contri.npy")
    gas_id = {'H$_2$O': 0, 'CH$_4$': 1, 'CO': 2, 'CO$_2$': 3, 'NH$_3$': 4}
    if gases is None:
        gases = label
    for idx, gas in enumerate(gases):
        plt.figure(figsize=(10, 8))
        if gas in ['H$_2$O', 'CH$_4$', 'CO', 'CO$_2$', 'NH$_3$']:
            ax1 = plt.subplot(2, 1, 1)
        else:
            ax1 = plt.subplot(1, 1, 1)
        if gas == 'R$_p$':
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
        if gas in ['H$_2$O', 'CH$_4$', 'CO', 'CO$_2$', 'NH$_3$']:
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
