import numpy as np
import keras
import sys
import os
from tensorflow import keras
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from .plotting import plot_compare_truth
from .utils import project_back
from .architecture import CNN_model_hp, LSTM_model, MLP_model
from kerastuner.tuners import RandomSearch


class Network():
    def __init__(self, param_length, spectrum_length, config):
        self.param_length = param_length
        self.spectrum_length = spectrum_length
        self.config = config

    def compile_model(self, hp):
        Available_model = np.array([self.config['training']['useCNN'],
                                    self.config['training']['useMLP'],
                                    self.config['training']['useLSTM']])
        if Available_model.sum() != 1:
            print("Please select an architecture")
            sys.exit()
        Architectures = np.array([CNN_model_hp, MLP_model, LSTM_model])
        idx = np.argwhere(Available_model == True)[0][0]
        print(np.argwhere(Available_model == True))
        build_model = Architectures[idx]
        sequence, param, decision_layer = build_model(
            param_length=self.param_length, spectrum_length=self.spectrum_length, config=self.config, hp=hp)

        self.model = Model(inputs=sequence, outputs=decision_layer)
        self.model.compile(loss=self.config['training']['lossFn'],
                           optimizer=keras.optimizers.Adam(lr=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4]), decay=10**self.config['training']['decay']), metrics=['mse'])
        self.model.summary()
        return self.model

    def train_model(self, X_train, y_train, X_valid, y_valid, epochs=30, lr=0.001, batch_size=64, checkpoint_dir='./', cv_order=0):
        print('training begins')
        # make sure no prior graph existss
        keras.backend.clear_session()

        # create log folder and checkpoint folder
        os.makedirs(os.path.join(checkpoint_dir, 'history'), exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir, 'ckt'), exist_ok=True)

        # callbacks
        # initialise log file.
        csv_logger = CSVLogger(os.path.join(
            checkpoint_dir, 'history/training_{}.log'.format(cv_order)))
        # initialise model checkpoint
        model_cktpt = ModelCheckpoint(os.path.join(checkpoint_dir, f'ckt/checkpt_{cv_order}.h5'),
                                      monitor='val_loss',
                                      save_best_only=True,
                                      mode='min',
                                      verbose=0,
                                      period=1)
        callbacks = [model_cktpt, csv_logger]

        # self.compile_model(lr=lr)
        tuner = RandomSearch(
            self.compile_model,
            objective='val_loss',
            max_trials=5,
            executions_per_trial=3,
            directory='my_dir',
            project_name='helloworld')
        tuner.search_space_summary()
        # display the network
        if self.config['general']['displayNet']:
            plot_model(self.model, to_file=os.path.join(
                checkpoint_dir, 'model.png'), show_shapes=True)

        # ensure they have the right shape
        if isinstance(X_train, list):
            X_train[0] = X_train[0].reshape(-1, self.spectrum_length, 1)
            X_valid[0] = X_valid[0].reshape(-1, self.spectrum_length, 1)
        else:

            X_train = X_train.reshape(-1, self.spectrum_length, 1)
            X_valid = X_valid.reshape(-1, self.spectrum_length, 1)

        # trainings
        # self.model.fit(X_train, y_train,
        #                batch_size=batch_size,
        #                epochs=epochs,
        #                verbose=1,
        #                validation_data=(X_valid, y_valid),
        #                shuffle=True,
        #                callbacks=callbacks)
        tuner.search(X_train, y_train,
                     epochs=30,
                     validation_data=(X_valid, y_valid))
        # score = self.model.evaluate(X_valid, y_valid, verbose=0)
        return None

    def load_model(self, checkpoint_dir_path):
        self.model = load_model(checkpoint_dir_path)
        return self.model

    def predict_result(self, x_test):
        if isinstance(x_test, list):
            x_test[0] = x_test[0].reshape(-1, self.spectrum_length, 1)
        else:
            x_test = x_test.reshape(-1, self.spectrum_length, 1)
        prediction = self.model.predict(x_test)
        return prediction

    def produce_result(self, std_x_test, y_test, param_mean, param_std, checkpoint_dir, order=0):
        os.makedirs(os.path.join(checkpoint_dir, 'results'), exist_ok=True)
        y_predict = self.predict_result(std_x_test)
        y_predict_org = project_back(y_predict, param_mean, param_std)
        y_test_org = project_back(y_test, param_mean, param_std)

        plot_compare_truth(y_test_org=y_test_org,
                           y_predict_org=y_predict_org,
                           checkpoint_dir=checkpoint_dir,
                           order=order,
                           scale=None,
                           chosen_gas=None, alpha=0.4)
