import numpy as np
import pandas as pd


def shuffle_data(input_param, predict_param, spectrum):
    index = list(range(len(input_param)))
    np.random.shuffle(index)
    input_param, predict_param, spectrum = input_param[index], predict_param[index], spectrum[index]
    return input_param, predict_param, spectrum


def param_batch_scaler(param, col_mean=None, col_std=None):
    log_param = np.log10(param[:, :6])
    logit_param = param[:, 6:]

    all_param = np.column_stack([log_param, logit_param])
    if col_mean is None:
        col_mean = np.mean(all_param, axis=0)
        col_std = np.std(all_param, axis=0)
    standard_data = (all_param - col_mean) / col_std

    return standard_data, col_mean, col_std


def gather_gas_MSE(df, y_test, y_predict, mean, std, chosen_gas):
    y_test_org = project_back(y_test, mean, std)
    y_predict_org = project_back(y_predict, mean, std)
    MSE = np.mean(np.square(y_test_org-y_predict_org), axis=0)

    temp_dict = {}
    for idx, gas in enumerate(chosen_gas):
        temp_dict[gas] = MSE[idx]
    df = df.append(temp_dict, ignore_index=True)
    return df


def compute_randbin_MSE(model, y_test, full_spectrum, y_data_mean=0, y_data_std=0,
                        chosen_gas=None, no_spectra=200, repeat=100, x_mean=0, x_std=0, additional_input=None, abundance=[-4, -3, ]):
    gases = chosen_gas
    y_test_org = project_back(y_test, y_data_mean, y_data_std)
    seed_x_test = full_spectrum[:, 0, :]
    accumulated_std = np.zeros((len(chosen_gas), no_spectra, repeat, 52))
    for idx, gas in enumerate(gases):

        if gas in ['H2O', 'CH4', 'CO', 'CO2', 'NH3']:
            index = (10 ** y_test_org.T[idx] > 10 ** abundance[0]
                     ) & (10 ** y_test_org.T[idx] < 10 ** abundance[1])
            x_test_selected = seed_x_test[index]
        elif gas == ['Rp']:
            index = np.where(y_test_org.T[idx] < 1 * 69911000)
            x_test_selected = seed_x_test[index]
        elif gas == ['Tp']:
            x_test_selected = seed_x_test.copy()
        for i in range(no_spectra):
            demo_spec = x_test_selected[i].copy()
            demo_stand = standardise(demo_spec, x_mean, x_std)
            # output_demo = model.predict(demo_stand.reshape(-1, 52, 1))
            if additional_input is None:
                output_demo = model.predict(demo_stand.reshape(-1, 52, 1))
            else:
                output_demo = model.predict(
                    [demo_stand.reshape(-1, 52, 1), additional_input])
            output_demo_org = project_back(
                output_demo, y_data_mean, y_data_std)[0]
            predicted_abundance = output_demo_org[idx]

            for k in range(repeat):
                att_spec = x_test_selected[i].copy()
                error = full_spectrum[i, 1, :]
                spectrum_length = len(att_spec)
                random_index = np.arange(len(att_spec))
                np.random.shuffle(random_index)
                curr = 0
                selected_bins = len(att_spec)
                while selected_bins >= 2:
                    att_spec = x_test_selected[i].copy()
                    new_selected = int(np.ceil(selected_bins/2))
                    picked_index = random_index[curr:curr+new_selected]
                    curr += new_selected
                    selected_bins = new_selected

                    shifted_pt = np.random.normal(
                        loc=att_spec[picked_index], scale=error[picked_index])
                    # shifted_pt = np.random.normal(loc=att_spec[picked_index], scale=mean_error)
                    att_spec[picked_index] = shifted_pt
                    atten_stand = standardise(att_spec, x_mean, x_std)

                    if additional_input is None:
                        output_atten = model.predict(
                            atten_stand.reshape(-1, 52, 1))
                    else:
                        output_atten = model.predict(
                            [atten_stand.reshape(-1, 52, 1), additional_input])
                    output_atten_org = project_back(
                        output_atten, y_data_mean, y_data_std)[0]
                    accumulated_std[idx, i, k,
                                    picked_index] = output_atten_org[idx]
            accumulated_std[idx, i, :, :] = np.square(
                accumulated_std[idx, i, :, :] - predicted_abundance)

    mean_MSE = np.mean(accumulated_std, axis=(1, 2))
    return mean_MSE


def compute_randbin_norm_MSE(model, y_test, full_spectrum, y_data_mean=0, y_data_std=0,
                             chosen_gas=None, no_spectra=200, repeat=100, x_mean=0, x_std=0, additional_input=None, abundance=[-6, -3]):
    gases = chosen_gas
    y_test_org = project_back(y_test, y_data_mean, y_data_std)
    seed_x_test = full_spectrum[:, 0, :]
    accumulated_std = np.zeros((len(chosen_gas), no_spectra, repeat, 52))
    # looking at special case when H2O is low

    selected_y = (y_test_org[:, 0] > abundance[0]) & (y_test_org[:, 1] > abundance[0]) & (
        y_test_org[:, 2] > abundance[0]) & (y_test_org[:, 3] > abundance[0]) & (y_test_org[:, 4] > abundance[0])
    x_test_selected = seed_x_test[selected_y].copy()
    for i in range(no_spectra):
        demo_spec = x_test_selected[i].copy()
        demo_stand = standardise(demo_spec, x_mean, x_std)
        # output_demo = model.predict(demo_stand.reshape(-1, 52, 1))
        if additional_input is None:
            output_demo = model.predict(demo_stand.reshape(-1, 52, 1))
        else:
            output_demo = model.predict(
                [demo_stand.reshape(-1, 52, 1), additional_input])
        output_demo_org = output_demo[0]
        predicted_abundance = output_demo_org

        for k in range(repeat):
            att_spec = x_test_selected[i].copy()
            error = full_spectrum[i, 1, :]
            spectrum_length = len(att_spec)
            random_index = np.arange(len(att_spec))
            np.random.shuffle(random_index)
            curr = 0
            selected_bins = len(att_spec)
            while selected_bins >= 2:
                att_spec = x_test_selected[i].copy()
                new_selected = int(np.ceil(selected_bins/2))
                picked_index = random_index[curr:curr+new_selected]
                curr += new_selected
                selected_bins = new_selected

                shifted_pt = np.random.normal(
                    loc=att_spec[picked_index], scale=error[picked_index])
                # shifted_pt = np.random.normal(loc=att_spec[picked_index], scale=mean_error)
                att_spec[picked_index] = shifted_pt
                atten_stand = standardise(att_spec, x_mean, x_std)

                if additional_input is None:
                    output_atten = model.predict(
                        atten_stand.reshape(-1, 52, 1))
                else:
                    output_atten = model.predict(
                        [atten_stand.reshape(-1, 52, 1), additional_input])
                output_atten_org = output_atten[0]
#                 print(picked_index.shape,i,k)
                accumulated_std[:, i, k,
                                picked_index] = output_atten_org[:, np.newaxis]
        accumulated_std[:, i, :, :] = np.square(
            accumulated_std[:, i, :, :] - predicted_abundance[:, np.newaxis, np.newaxis])

    mean_MSE = np.mean(accumulated_std, axis=(1, 2))
    return mean_MSE


def compute_bias_var_sensi(model, y_test, full_spectrum, y_data_mean=0, y_data_std=0,
                           chosen_gas=None, no_spectra=200, repeat=100, x_mean=0, x_std=0, additional_input=None, slider=3, abundance=[-4, -3]):

    gases = chosen_gas
    y_test_org = project_back(y_test, y_data_mean, y_data_std)
    seed_x_test = full_spectrum[:, 0, :]
    accumulated_std = np.zeros((len(chosen_gas), no_spectra, repeat, 52))
    for idx, gas in enumerate(gases):

        if gas in ['H2O', 'CH4', 'CO', 'CO2', 'NH3']:
            index = (10 ** y_test_org.T[idx] > 10**abundance[0]
                     ) & (10 ** y_test_org.T[idx] < 10**abundance[1])
            x_test_selected = seed_x_test[index]
        elif gas in ['Rp', ]:
            index = np.where(y_test_org.T[idx] < 1 * 69911000)
            x_test_selected = seed_x_test[index]
        elif gas in ['Mp', 'Tp', 'cloud']:
            x_test_selected = seed_x_test.copy()
        for i in range(no_spectra):
            demo_spec = x_test_selected[i].copy()
            demo_stand = standardise(demo_spec, x_mean, x_std)
            if additional_input is None:
                output_demo = model.predict(demo_stand.reshape(-1, 52, 1))
            else:
                output_demo = model.predict(
                    [demo_stand.reshape(-1, 52, 1), additional_input])
            output_demo_org = project_back(
                output_demo, y_data_mean, y_data_std)[0]
            predicted_abundance = output_demo_org[idx]

            for k in range(repeat):
                att_spec = x_test_selected[i].copy()
                error = full_spectrum[i, 1, :]
                while True:
                    random_num = np.random.randint(
                        low=0, high=52-slider+1, size=2)
                    if (random_num[1] - random_num[0]) > slider:
                        break
                    else:
                        pass
                index_list = np.arange(0, len(att_spec))
                selected_index = np.append(np.arange(random_num[0], random_num[0] + slider),
                                           np.arange(random_num[1], random_num[1] + slider))
                # selected_index[selected_index > 51] -= 52
                for pts in selected_index:
                    shifted_pt = np.random.normal(
                        loc=att_spec[pts], scale=error[pts], size=1)
                    att_spec[pts] = shifted_pt
                atten_stand = standardise(att_spec, x_mean, x_std)
                if additional_input is None:
                    output_atten = model.predict(
                        atten_stand.reshape(-1, 52, 1))
                else:
                    output_atten = model.predict(
                        [atten_stand.reshape(-1, 52, 1), additional_input])
                output_atten_org = project_back(
                    output_atten, y_data_mean, y_data_std)[0]
                accumulated_std[idx, i, k,
                                selected_index] = output_atten_org[idx]
                unselected = np.setdiff1d(index_list, selected_index)
                accumulated_std[idx, i, k, unselected] = predicted_abundance
            accumulated_std[idx, i, :, :] = np.abs(
                accumulated_std[idx, i, :, :] - predicted_abundance)
    accumulated_std = np.where(np.isclose(
        accumulated_std, 0), np.nan, accumulated_std)
    total_std = np.nanstd(accumulated_std, axis=(1, 2))
    total_mean = np.nanmean(accumulated_std, axis=(1, 2))
    return total_std, total_mean
