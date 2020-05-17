import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import Callback
from keras.models import load_model
from keras.backend import gradients
from matplotlib import gridspec
import neptune
import os
import datetime

project = neptune.init(api_token=os.environ.get('NEPTUNE_API_TOKEN'), project_qualified_name='harryandrews1/M5_Challenge')

PARAMS = {'neurons': 40,
          'lr': 0.0001,
          'dropout': 0.2,
          'batch_size': 64,
          'optimizer': 'adam',
          'loss': 'mean_squared_error',
          'metrics': ['accuracy'],
          'n_epochs': 100,
          }

def scale_data(x, min_x, max_x):
    x_scaled = np.empty(x.shape)
    n_entries = x.shape[0]
    n_filts = x.shape[1]

    for i in range(0, n_entries):
        for j in range(0, n_filts):
            val = x[i, j]
            if val > max_x:
                x_scaled[i, j] = 1
            elif val < min_x:
                x_scaled[i, j] = 0
            else:
                #x_scaled[i, j] = (2 * (val - min_x) / (max_x - min_x)) - 1
                x_scaled[i, j] = (val - min_x) / (max_x - min_x)

    return x_scaled

class NeptuneMonitor(Callback):
    def __init__(self, neptune_experiment, n_batch):
        super().__init__()
        self.exp = neptune_experiment
        self.n = n_batch
        self.current_epoch = 0

    def on_batch_end(self, batch, logs=None):
        x = (self.current_epoch * self.n) + batch
        self.exp.send_metric(channel_name='batch end loss', x=x, y=logs['loss'])
        #self.exp.send_metric(channel_name='batch end val_loss', x=x, y=logs['val_loss'])
    def on_epoch_end(self, epoch, logs=None):
        self.exp.send_metric('epoch end loss', logs['loss'])
        #self.exp.send_metric('epoch end val_loss', logs['val_loss'])
        #innovative_metric = logs['acc'] - 2 * logs['loss']
        #self.exp.send_metric(channel_name='innovative_metric', x=epoch, y=innovative_metric)

        msg_loss = 'End of epoch {}, categorical crossentropy loss is {:.4f}'.format(epoch, logs['loss'])
        self.exp.send_text(channel_name='loss information', x=epoch, y=msg_loss)

        self.current_epoch += 1


class ModelHandler():
    def __init__(self):
        self.model = None
        self.description = {}

    def load_model(self, str_model, loss):
        self.model = load_model(str_model)
        self.model.compile(optimizer=PARAMS['optimizer'],
                           loss=loss,
                           metrics=PARAMS['metrics'])

        self.description['type'] = 'LSTM + DNN'
        self.description['loss'] = loss
        self.description['config'] = self.model.get_config()
        self.description['n_epochs'] = PARAMS['n_epochs']
        self.description['batch_size'] = PARAMS['batch_size']
        self.description['snr'] = snr
        self.description['training_entries'] = x_train.shape[0]

        self.model.summary()
        return self.model

    def setupLSTM(self, x_train, loss):
        optimiser = keras.optimizers.Adam(learning_rate=PARAMS['lr'], clipnorm=1, clipvalue=0.5)
        self.model = keras.models.Sequential([
            keras.layers.LSTM(PARAMS['neurons'], input_shape=(x_train.shape[1], x_train.shape[2]),
                              activation=keras.activations.relu, return_sequences=True),
            keras.layers.LSTM(PARAMS['neurons'], activation=keras.activations.relu, return_sequences=False),
            #keras.layers.Dense(PARAMS['neurons'], activation=keras.activations.relu),
            keras.layers.Dense(PARAMS['neurons'], activation=keras.activations.relu),
            keras.layers.Dense(PARAMS['neurons'], activation=keras.activations.relu),
            keras.layers.Dropout(PARAMS['dropout']),
            keras.layers.Dense(1, activation=keras.activations.sigmoid)
        ])
        self.description['type'] = 'LSTM + DNN'
        self.description['loss'] = loss
        self.description['config'] = self.model.get_config()
        self.description['n_epochs'] = PARAMS['n_epochs']
        self.description['batch_size'] = PARAMS['batch_size']
        self.description['snr'] = snr
        self.description['training_entries'] = x_train.shape[0]

        self.model.compile(optimizer=optimiser,
                      loss=loss,
                      metrics=PARAMS['metrics'])
        log_dir = "/Users/hxa138/Kaggle/M5_forecastingchallenge/Harry/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        self.model.summary()
        return self.model

    def fitModel(self, x_train, targets, x_val, targets_val):
        with project.create_experiment(name='Binary Speech Classifcation Task',
                                       params=PARAMS,
                                       description='%s network trained with %s loss for %d epochs with a batch size of %d on %d entries with a SNR of %ddB' % (
                                               self.description['type'], self.description['loss'],
                                               self.description['n_epochs'],
                                               self.description['batch_size'], self.description['training_entries'],
                                               self.description['snr'])) as npt_exp:
            npt_exp.set_property('n_batches', n_batches)
            npt_exp.set_property('n_epochs', self.description['n_epochs'])
            npt_exp.set_property('snr', self.description['snr'])
            npt_exp.set_property('network type', self.description['type'])
            npt_exp.set_property('loss', self.description['loss'])
            npt_exp.set_property('learning rate', PARAMS['lr'])
            npt_exp.set_property('neurons per layer', PARAMS['neurons'])

            self.history = self.model.fit(x_train, targets,
                                          validation_data=[x_val, targets_val],
                                          epochs=PARAMS['n_epochs'],
                                          batch_size=PARAMS['batch_size'],
                                          callbacks=[NeptuneMonitor(npt_exp, n_batches),
                                                     self.tensorboard_callback],
                                          verbose=2)

            loss, accuracy = self.model.evaluate(x_train, targets)
            print('accuracy = %.2f  loss = %.2f' % (accuracy, loss))
            self.description['accuracy'] = accuracy
            self.description['loss_score'] = loss
            npt_exp.set_property('accuracy', self.description['accuracy'])
            npt_exp.set_property('loss score', self.description['loss_score'])

            loss_val, accuracy_val = self.model.evaluate(x_val, targets_val)
            print('validation accuracy = %.2f  validation loss = %.2f' % (accuracy_val, loss_val))
            self.description['validation accuracy'] = accuracy_val
            self.description['validation loss_score'] = loss_val
            npt_exp.set_property('validation accuracy', self.description['validation accuracy'])
            npt_exp.set_property('validation loss score', self.description['validation loss_score'])
            npt_exp.set_property('clipped gradients', True)
        return self.history

    def getModel(self):
        return self.model

    def prepare_data(raw_data, y, timesteps, lookahead):
        n_frames = raw_data.shape[0]
        n_bands = raw_data.shape[1]
        data = np.empty((n_frames - timesteps + 1, timesteps, n_bands))
        targets = np.empty((n_frames - timesteps + 1, 1))
        dnn_targets = np.empty((n_frames - timesteps + 1, 1))
        scaled_data = scale_data(raw_data, -50.0, 20.0)
        for i in range(timesteps, n_frames + 1):
            entry = np.empty((timesteps, n_bands))
            for j in range(timesteps):
                entry[timesteps - j - 1, :] = scaled_data[i - j - 1, :]
            data[i - timesteps, :, :] = entry
            # targets[i - timesteps] = np.mean(entry[timesteps-2:timesteps-1])
            # targets[i - timesteps] = np.mean(entry[timesteps-2])
            if entry[timesteps - 1, 0] > 50:
                # targets[i - timesteps] = np.mean(entry[timesteps-1])
                targets[i - timesteps] = y[i - lookahead - 1]
                dnn_targets[i - timesteps] = y[i - 1]
            else:
                # targets[i - timesteps] = np.mean(entry[timesteps-1])
                targets[i - timesteps] = y[i - lookahead - 1]
                dnn_targets[i - timesteps] = y[i - 1]
        return data, targets, dnn_targets

if __name__ == '__main__':
    num_bands = 40
    timesteps = 20
    lookahead = 5

    d1_col = 6              # First day column
    n_days = 100            # Number of days to analyse

    filename = '../data/'   # Data directory

    state = 'CA'            # Data subdivision to analyse, leave blank study entire dataset

    if state:
        filename += '%s/%s_data.csv' % (state, state)
    else:
        filename += 'sales_train_validation.csv'

    df = pd.read_csv(filename, index_col=0)                 # Read data
    calendar_df = pd.read_csv('../data/calendar.csv')       # Read the calendar csv for notable dates

    #

    print(df.columns)

