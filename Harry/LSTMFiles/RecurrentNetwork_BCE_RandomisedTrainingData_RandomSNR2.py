import numpy as np
import matplotlib.pyplot as plt
import soundfile
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
#import keras
from keras.callbacks import Callback
from keras.models import load_model
from keras.backend import gradients
from matplotlib import gridspec
import neptune
import os
import datetime

project = neptune.init(api_token=os.environ.get('NEPTUNE_API_TOKEN'), project_qualified_name='harryandrews1/DeepGreen')

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
        log_dir = "E:\\work\\Research\\DNN\\logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        self.model.summary()
        return self.model

    def setupDNN(self, input, loss):
        self.model = keras.models.Sequential([
            keras.layers.Dense(PARAMS['neurons'], input_dim=(input.shape[1]), activation=keras.activations.relu),
            keras.layers.Dense(PARAMS['neurons'], activation=keras.activations.relu),
            keras.layers.Dense(PARAMS['neurons'], activation=keras.activations.relu),
            keras.layers.Dense(PARAMS['neurons'], activation=keras.activations.relu),
            keras.layers.Dropout(PARAMS['dropout']),
            keras.layers.Dense(1, activation=keras.activations.sigmoid)
        ])
        self.description['type'] = 'DNN'
        self.description['loss'] = loss
        self.description['config'] = self.model.get_config()
        self.description['n_epochs'] = PARAMS['n_epochs']
        self.description['batch_size'] = PARAMS['batch_size']
        self.description['snr'] = snr
        self.description['training_entries'] = x_train.shape[0]

        self.model.compile(optimizer=PARAMS['optimizer'],
                      loss=loss,
                      metrics=PARAMS['metrics'])

        log_dir = "E:\\work\\Research\\DNN\\logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.model.summary()
        return self.model

    def fitModel(self, x_train, targets, x_val, targets_val):
        with project.create_experiment(name='Binary Speech Classifcation Task',
                                       params=PARAMS,
                                       description='%s network trained with %s loss for %d epochs with a batch size of %d on %d entries with a SNR of %ddB' % (
                                       self.description['type'], self.description['loss'], self.description['n_epochs'],
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
    data = np.empty((n_frames-timesteps+1, timesteps, n_bands))
    targets = np.empty((n_frames-timesteps+1, 1))
    dnn_targets = np.empty((n_frames-timesteps+1, 1))
    scaled_data = scale_data(raw_data, -50.0, 20.0)
    for i in range(timesteps, n_frames+1):
        entry = np.empty((timesteps, n_bands))
        for j in range(timesteps):
            entry[timesteps - j - 1, :] = scaled_data[i-j-1, :]
        data[i - timesteps, :, :] = entry
        #targets[i - timesteps] = np.mean(entry[timesteps-2:timesteps-1])
        #targets[i - timesteps] = np.mean(entry[timesteps-2])
        if entry[timesteps-1, 0] > 50:
            #targets[i - timesteps] = np.mean(entry[timesteps-1])
            targets[i - timesteps] = y[i - lookahead - 1]
            dnn_targets[i - timesteps] = y[i-1]
        else:
            #targets[i - timesteps] = np.mean(entry[timesteps-1])
            targets[i - timesteps] = y[i - lookahead - 1]
            dnn_targets[i - timesteps] = y[i - 1]
    return data, targets, dnn_targets

if __name__ == '__main__':
    num_bands = 40
    timesteps = 20
    lookahead = 5
    #input = np.random.randint(0, 100, [num_entries, num_bands])
    #val_input = np.random.randint(0, 100, [num_entries, num_bands])
    #input = np.random.random([num_entries, num_bands])
    #val_input = np.random.random([num_entries, num_bands])
    cols = []
    for i in range(15, num_bands):
        cols.append('Filter %d' % i)
    cols.append('Target')
    snr = 20
    noise = 99
    input_frames = []
    x_train_dnn = x_val_dnn = np.empty((0, len(cols)-1))
    x_train = x_val = np.empty((0, timesteps, len(cols)-1))
    targets = targets_val = dnn_targets = dnn_targets_val = np.empty((0, 1))
    speech_input_dir = 'E:\\TrainingData\\csvs\\random\\'
    noise_input_dir = 'E:\\TrainingData\\csvs\\noise\\'
    wav_dir = 'E:\\TrainingData\\wavs\\random\\'
    for i in range(noise):
        track = np.random.randint(1, 18)
        track_val = track
        while track_val == track:
            track_val = np.random.randint(1, 18)
        with os.scandir(speech_input_dir) as entries:
            for entry in entries:
                if entry.is_file():
                    if 'Track%d.' % track in entry.name and 'Noisen%d-' % i in entry.name:
                        print('data input = %s' % entry.name)
                        df = pd.read_csv(entry, nrows=2000, index_col=False, usecols=cols)
                        data = df.iloc[0:, 0:len(cols)-1]
                        y = df.iloc[0:, len(cols)-1]
                        data = np.array(data)
                        y = np.array(y)
                        x_train_tmp, targets_tmp, dnn_targets_tmp = prepare_data(data, y, timesteps=timesteps, lookahead=lookahead)
                        rand_scale = np.random.random()
                        x_train_tmp = x_train_tmp * rand_scale

                        x_train = np.concatenate((x_train, x_train_tmp), axis=0)
                        targets = np.concatenate((targets, targets_tmp), axis=0)
                        dnn_targets = np.concatenate((dnn_targets, dnn_targets_tmp), axis=0)
                        x_train_dnn = np.concatenate((x_train_dnn, rand_scale * scale_data(data[timesteps-1:], -50.0, 20.0)))

                    elif 'Track%d.' % track_val in entry.name and 'Noisen%d-' % i in entry.name:
                        print('val input = %s' % entry.name)
                        val_df = pd.read_csv(entry, nrows=2000, index_col=False, usecols=cols)
                        val_data = val_df.iloc[0:, 0:len(cols)-1]
                        val_y = val_df.iloc[0:, len(cols)-1]
                        val_data = np.array(val_data)
                        val_y = np.array(val_y)
                        x_val_tmp, targets_val_tmp, dnn_targets_val_tmp = prepare_data(val_data, val_y, timesteps=timesteps,
                                                                                       lookahead=lookahead)
                        rand_scale = np.random.random()
                        x_val_tmp = x_val_tmp * rand_scale

                        x_val = np.concatenate((x_val, x_val_tmp), axis=0)
                        targets_val = np.concatenate((targets_val, targets_val_tmp), axis=0)
                        dnn_targets_val = np.concatenate((dnn_targets_val, dnn_targets_val_tmp), axis=0)
                        x_val_dnn = np.concatenate((x_val_dnn, rand_scale * scale_data(val_data[timesteps-1:], -50.0, 20.0)))

            with os.scandir(noise_input_dir) as entries:
                for entry in entries:
                    if entry.is_file():
                        if 'Track%d.' % track in entry.name and 'Noisen%d-' % i in entry.name:
                            print('data input = %s' % entry.name)
                            df = pd.read_csv(entry, nrows=2000, index_col=False, usecols=cols)
                            data = df.iloc[0:, 0:len(cols)-1]
                            y = df.iloc[0:, len(cols)-1]
                            data = np.array(data)
                            y = np.array(y)
                            x_train_tmp, targets_tmp, dnn_targets_tmp = prepare_data(data, y, timesteps=timesteps,
                                                                                     lookahead=lookahead)

                            x_train = np.concatenate((x_train, x_train_tmp), axis=0)
                            targets = np.concatenate((targets, targets_tmp), axis=0)
                            dnn_targets = np.concatenate((dnn_targets, dnn_targets_tmp), axis=0)
                            x_train_dnn = np.concatenate(
                                (x_train_dnn, rand_scale * scale_data(data[timesteps - 1:], -50.0, 20.0)))

                        elif 'Track%d.' % track_val in entry.name and 'Noisen%d-' % i in entry.name:
                            print('data input = %s' % entry.name)
                            val_df = pd.read_csv(entry, nrows=2000, index_col=False, usecols=cols)
                            val_data = val_df.iloc[0:, 0:len(cols)-1]
                            val_y = val_df.iloc[0:, len(cols)-1]
                            val_data = np.array(val_data)
                            val_y = np.array(val_y)
                            x_val_tmp, targets_val_tmp, dnn_targets_val_tmp = prepare_data(val_data, val_y,
                                                                                           timesteps=timesteps,
                                                                                           lookahead=lookahead)
                            x_val = np.concatenate((x_val, x_val_tmp), axis=0)
                            targets_val = np.concatenate((targets_val, targets_val_tmp), axis=0)
                            dnn_targets_val = np.concatenate((dnn_targets_val, dnn_targets_val_tmp), axis=0)
                            x_val_dnn = np.concatenate(
                                (x_val_dnn, rand_scale * scale_data(val_data[timesteps - 1:], -50.0, 20.0)))

    x_train_shuff, x_val_shuff,  targets_shuff, targets_val_shuff = train_test_split(x_train, targets, shuffle=True)
    x_train_shuff_dnn, x_val_shuff_dnn,  targets_shuff_dnn, targets_val_shuff_dnn = train_test_split(x_train_dnn, dnn_targets, shuffle=True)
    print(x_train.shape)
    print(targets.shape)
    print(x_val.shape)
    print(targets_val.shape)
    run = 2
    load_pretrained_model = False
    n_batches = x_train.shape[0] // PARAMS['batch_size'] + 1

    DNN_MSE = ModelHandler()
    if load_pretrained_model:
        DNN_MSE.load_model(
            'C:\\Users\\Harry Andrews\\Documents\\Projects\\DNN\\models\\DNN_MSE_%dEpochs_%dLookahead_%d.h5' % (
            timesteps, lookahead, run), 'mean_squared_error')
    else:
        model_DNN_MSE = DNN_MSE.setupDNN(x_train_shuff_dnn, 'mean_squared_error')
    history_DNN_MSE = DNN_MSE.fitModel(x_train_shuff_dnn, targets_shuff_dnn, x_val_shuff_dnn, targets_val_shuff_dnn)
    model_DNN_MSE = DNN_MSE.getModel()
    predictions_DNN_MSE = model_DNN_MSE.predict(x_train_dnn)
    predictions_val_DNN_MSE = model_DNN_MSE.predict(x_val_dnn)
    model_DNN_MSE.save(
        'C:\\Users\\Harry\\Documents\\DNN_Project\\models\\DNN_models\\DNN_MSE_%dEpochs_%dLookahead_%d.h5' % (
        timesteps, lookahead, run))

    LSTM_MSE = ModelHandler()
    if load_pretrained_model:
        LSTM_MSE.load_model('C:\\Users\\Harry Andrews\\Documents\\Projects\\DNN\\models\\LSTM_MSE_%dEpochs_%dLookahead_%d.h5' % (timesteps, lookahead, run), 'mean_squared_error')
    else:
        model_LSTM_MSE = LSTM_MSE.setupLSTM(x_train, 'mean_squared_error')
    history_LSTM_MSE = LSTM_MSE.fitModel(x_train_shuff, targets_shuff, x_val_shuff, targets_val_shuff)
    model_LSTM_MSE = LSTM_MSE.getModel()
    predictions_LSTM_MSE = model_LSTM_MSE.predict(x_train)
    predictions_val_LSTM_MSE = model_LSTM_MSE.predict(x_val)
    model_LSTM_MSE.save('C:\\Users\\Harry\\Documents\\DNN_Project\\models\\LSTM_models\\LSTM_MSE_%dEpochs_%dLookahead_%d.h5' % (timesteps, lookahead, run))



    gradients = gradients(model_LSTM_MSE.output, model_LSTM_MSE.input)  # Gradient of output wrt the input of the model (Tensor)
    print(gradients)
    print(gradients[0])
    plt.figure(1)
    plt.plot(targets, label='Targets')
    plt.plot(predictions_LSTM_MSE, label='LSTM MSE')
    plt.legend(loc='upper right')

    plt.figure(2)
    plt.plot(targets_val, label='Targets')
    plt.plot(predictions_val_LSTM_MSE, label='LSTM MSE')
    plt.legend(loc='upper right')

    plt.figure(3)
    plt.plot(history_LSTM_MSE.history['loss'], label='LSTM MSE')
    plt.plot(history_LSTM_MSE.history['val_loss'], label='LSTM MSE (Val)')
    plt.legend(loc='upper right')

    plt.figure(4)
    plt.plot(history_LSTM_MSE.history['accuracy'], label='LSTM MSE')
    plt.plot(history_LSTM_MSE.history['val_accuracy'], label='LSTM MSE (Val)')
    plt.legend(loc='lower right')

    plt.figure(5)
    plt.subplot(2, 1, 1)
    plt.plot(x_train_dnn)
    plt.subplot(2, 1, 2)
    plt.plot(np.concatenate((np.zeros((4, 1)), targets)))

    plt.show()