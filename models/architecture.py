
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, Conv1D, MaxPooling1D, Input, LeakyReLU, LSTM, Bidirectional, Concatenate


def CNN_model(spectrum_length, param_length, config):
    # input image dimensions
    input_shape = (spectrum_length, 1)
    param_shape = (1,)
    # Start Neural Network

    sequence = Input(shape=input_shape)
    param = Input(shape=param_shape)
    x = sequence
    # CNN layer.
    for i in range(len(config['CNN']['layers'])):

        x = Conv1D(filters=config['CNN']['layers'][i],
                   kernel_size=config['CNN']['kernels'][i],
                   strides=config['CNN']['stride'][i],
                   padding='same')(x)
        if config['CNN']['batchNorm'][i] == 1:
            x = BatchNormalization(axis=-1)(x)
        if config['CNN']['activation'] == 'leaky_relu':
            x = LeakyReLU(0.2)(x)
        else:
            x = Activation(config['CNN']['activation'])(x)

        if config['CNN']['maxPool'][i] == 1:
            x = MaxPooling1D(pool_size=2)(x)

    flatten_x = Flatten()(x)
    if config['training']['extraInput']:
        add_param = param
        flatten_x = Concatenate(axis=-1)([flatten_x, add_param])
    dense_layer = Dense(
        config['CNN']['denseUnit'], activation='linear')(flatten_x)
    dense_layer = LeakyReLU(0.2)(dense_layer)
    dense_layer = Dropout(config['training']['dropRate'])(dense_layer)
    decision_layer = Dense(param_length, activation='linear')(dense_layer)
    return sequence, param, decision_layer


def LSTM_model(spectrum_length, param_length, config):
    # input image dimensions
    input_shape = (spectrum_length, 1)
    param_shape = (1,)
    # Start Neural Network

    sequence = Input(shape=input_shape)
    param = Input(shape=param_shape)
    x = sequence
    for i in range(len(config['LSTM']['layers'])):
        # remove the output seq on last layer. less hidden units.
        if i == len(config['LSTM']['layers'])-1:
            x = Bidirectional(
                LSTM(config['LSTM']['layers'][i], return_sequences=False))(x)
        else:
            x = Bidirectional(
                LSTM(config['LSTM']['layers'][i], return_sequences=True))(x)
    if config['training']['extraInput']:
        add_param = param
        x = Concatenate(axis=-1)([x, add_param])
    dense_layer = Dense(config['LSTM']['denseUnit'], activation='linear')(x)
    dense_layer = LeakyReLU(0.2)(dense_layer)
    dense_layer = Dropout(config['training']['dropRate'])(dense_layer)
    decision_layer = Dense(param_length, activation='linear')(dense_layer)
    return sequence, param, decision_layer


def MLP_model(spectrum_length, param_length, config):
    # input image dimensions
    input_shape = (spectrum_length, 1)
    param_shape = (1,)
    # Start Neural Network

    sequence = Input(shape=input_shape)
    param = Input(shape=param_shape)
    x = sequence
    activation = config['MLP']['activation']

    # Flatten the sequence for MLP layers
    x = Flatten()(x)

    # concatenate extra param to the sequence
    if config['training']['extraInput']:
        add_param = param
        x = Concatenate(axis=-1)([x, add_param])
    # MLP layer.
    for i in range(len(config['MLP']['layers'])):
        x = Dense(config['MLP']['layers'][i], activation=activation)(x)
    dense_layer = Dropout(config['training']['dropRate'])(x)
    decision_layer = Dense(param_length, activation='linear')(dense_layer)
    return sequence, param, decision_layer


def MLP_model_hp(spectrum_length, param_length, config, hp):
    # input image dimensions
    input_shape = (spectrum_length, 1)
    param_shape = (1,)
    # Start Neural Network

    sequence = Input(shape=input_shape)
    param = Input(shape=param_shape)
    x = sequence
    activation = config['MLP']['activation']

    # Flatten the sequence for MLP layers
    x = Flatten()(x)

    # concatenate extra param to the sequence
    if config['training']['extraInput']:
        add_param = param
        x = Concatenate(axis=-1)([x, add_param])
    # MLP layer.
    for i in range(hp.Int('num_layers', 2, 4)):
        x = Dense(hp.Int('units' + str(i), min_value=32,
                         max_value=256, step=32), activation=activation)(x)
    dense_layer = Dropout(hp.Choice('droprate', [0.1, 0.2, 0.3]))(x)
    decision_layer = Dense(param_length, activation='linear')(dense_layer)
    return sequence, param, decision_layer
