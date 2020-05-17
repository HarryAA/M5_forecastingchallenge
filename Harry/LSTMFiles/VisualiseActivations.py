import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import soundfile
import numpy as np
import pandas as pd
import theano
from tensorflow.keras.models import load_model
import keras
import math

def calc_performance(y, predictions):
    nonspeech_frames = speech_frames = accuracy = FAR = MR = 0
    for i in range(y.shape[0]):
        if y[i] == 0:
            nonspeech_frames += 1
            if predictions[i] < 0.5:
                accuracy += 1
            else:
                FAR += 1
        else:
            speech_frames += 1
            if predictions[i] >= 0.5:
                accuracy += 1
            else:
                MR += 1
    if nonspeech_frames != 0:
        FAR = FAR / nonspeech_frames
    if speech_frames != 0:
        MR = MR / speech_frames
    nonspeech_frames = nonspeech_frames / y.shape[0]
    speech_frames = speech_frames / y.shape[0]
    accuracy = accuracy / y.shape[0]

    return nonspeech_frames, speech_frames, accuracy, FAR, MR

def get_activations(model, layer, X_batch):
    get_activations = keras.backend.function([model.layers[0].input], model.layers[layer].output)
    activations = get_activations(X_batch) # same result as above
    return activations

def get_activations_lstm(model, layer, X_batch):
    lstm = model.layers[layer]
    get_activations = keras.backend.function([model.layers[0].input], outputs=[lstm.output])
    activations = get_activations(X_batch) # same result as above
    return activations

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

def prepare_data_LSTM(raw_data, y, timesteps, lookahead):
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
        targets[i - timesteps] = y[i - lookahead - 1]
        dnn_targets[i - timesteps] = y[i - 1]

    return data, targets, dnn_targets

def prepare_data_LSTM2(raw_data, y, timesteps, lookahead):
    n_frames = raw_data.shape[0]
    n_bands = raw_data.shape[1]
    data = np.empty((n_frames-timesteps+1, timesteps, n_bands))
    targets = np.empty((n_frames-timesteps+1, 1))
    dnn_targets = np.empty((n_frames-timesteps+1, 1))
    for i in range(timesteps, n_frames+1):
        entry = np.empty((timesteps, n_bands))
        for j in range(timesteps):
            entry[timesteps - j - 1, :] = raw_data[i-j-1, :]
        if math.isnan(entry.min()) or math.isinf(entry.min()):
            print(entry.min())
            print(entry.max())
            min_scale = -100.0
        else:
            min_scale = entry.min()
        if math.isnan(entry.max()) or math.isinf(entry.max()):
            print(entry.min())
            print(entry.max())
            max_scale = -100.0
        else:
            max_scale = entry.max()

        entry_scaled = scale_data(entry, min_scale, max_scale)
        data[i - timesteps, :, :] = entry_scaled
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

num_bands = 40
snr = 20
run = 11
timesteps = 20
lookahead = 5

slice_offset = 0
slice_size = 10000

track_range = range(1, 2)
noise_range = range(1, 2)

plot = True

cols = []
for i in range(15, num_bands):
    cols.append('Filter %d' % i)
cols.append('Target')

speech_input_dir = "E:\\TrainingData\\csvs\\%ddB\\" % snr
#str_model = ['C:\\Users\\Harry\\Documents\\DNN_Project\\models\\DNN_BCE_20Epochs_5Lookahead_8.h5',
             #'C:\\Users\\Harry\\Documents\\DNN_Project\\models\\DNN_MSE_20Epochs_5Lookahead_8.h5',
             #'C:\\Users\\Harry\\Documents\\DNN_Project\\models\\LSTM_BCE_20Epochs_5Lookahead_8.h5',
             #'C:\\Users\\Harry\\Documents\\DNN_Project\\models\\LSTM_MSE_20Epochs_5Lookahead_8.h5']
str_model = ['C:\\Users\\Harry\\Documents\\DNN_Project\\models\\LSTM_models\\LSTM_MSE_20Epochs_5Lookahead_6.h5']
#str_model = ['C:\\Users\\Harry\\Documents\\DNN_Project\\models\\DNN_BCE_20Epochs_5Lookahead_%d.h5' % run,
             #'C:\\Users\\Harry\\Documents\\DNN_Project\\models\\DNN_MSE_20Epochs_5Lookahead_%d.h5' % run,
             #'C:\\Users\\Harry\\Documents\\DNN_Project\\models\\LSTM_BCE_20Epochs_5Lookahead_%d.h5' % run,
             #'C:\\Users\\Harry\\Documents\\DNN_Project\\models\\LSTM_MSE_20Epochs_5Lookahead_%d.h5' % run]
subfigures = []
output_df = pd.DataFrame(columns=['Noise', 'Model', 'Loss', 'Accuracy', 'FAR', 'MR'])
for model_n in range(0, len(str_model)):
    model = load_model(str_model[model_n])
    for noise in noise_range:
        for track in track_range:
            df = pd.read_csv(speech_input_dir + 'mean_powers_NewTargetsLoadedNoisen%d-%ddBTrack%d.csv' % (noise, snr, track),  index_col=False, usecols=cols)
            #df = pd.read_csv(speech_input_dir + 'Noisen%d-%ddBTrack%d.csv' % (noise, snr, track),  index_col=False, usecols=cols)
            #df = pd.read_csv(speech_input_dir + 'mean_powers_Irish.csv', index_col=False, usecols=cols)
            #df = pd.read_csv('E:\\TrainingData\\mean_powers_Sennheiser.csv', index_col=False, usecols=cols)
            #df = pd.read_csv(speech_input_dir + 'LoadedNoisen2-5dBbook_09918_chp_0018_reader_09944_36.csv', index_col=False, usecols=cols)
            x_unscaled = df.iloc[0+slice_offset:slice_size+slice_offset, 0:len(cols)-1]
            x_unscaled = np.array(x_unscaled)
            y = df.iloc[0+slice_offset:slice_size+slice_offset, len(cols)-1]
            y = np.array(y)

            x_scaled_lstm, y_lstm, dnn_targets = prepare_data_LSTM2(x_unscaled, y, timesteps=timesteps, lookahead=lookahead)
            x_scaled = scale_data(x_unscaled, -50.0, 20.0)
            y = np.reshape(y, (y.shape[0], 1))
            print('number of layers = %d' % len(model.layers))
            if model_n > -1:
                activations_0 = get_activations(model, 0, x_scaled_lstm)
                activations_0 = np.reshape(activations_0[:, timesteps - 1, :],
                                           (activations_0.shape[0], activations_0.shape[2]))
                print(activations_0.shape)
                activations_1 = get_activations(model, 1, x_scaled_lstm)
                print(activations_1.shape)
                activations_2 = get_activations(model, 2, x_scaled_lstm)
                print(activations_2.shape)
                activations_3 = get_activations(model, 3, x_scaled_lstm)
                print(activations_3.shape)
                activations_4 = get_activations(model, 4, x_scaled_lstm)
                print(activations_4.shape)
                predictions = model.predict(x_scaled_lstm)
                nonspeech_frames, speech_frames, accuracy, FAR, MR = calc_performance(y_lstm, predictions)
                loss, accuracy = model.evaluate(x_scaled_lstm, y_lstm)
            else:
                activations_1 = get_activations(model, 1, x_scaled)
                activations_2 = get_activations(model, 2, x_scaled)
                activations_3 = get_activations(model, 3, x_scaled)
                activations_4 = get_activations(model, 4, x_scaled)
                predictions = model.predict(x_scaled)
                nonspeech_frames, speech_frames, accuracy, FAR, MR = calc_performance(y, predictions)
                loss, accuracy = model.evaluate(x_scaled, y)

            if plot:
                figure_tmp, axes = plt.subplots(7, 1, sharex=True, gridspec_kw={'height_ratios': [1, 1, 4, 4, 4, 4, 8]})
                #plt.xlim((5000, 7000))
                subfigures.append(figure_tmp)
                subfigures[model_n].subplots_adjust(hspace=0.1)
                if model_n > -1:
                    axes[0].imshow(np.transpose(y_lstm), aspect='auto', vmin=0.0, vmax=1.0)
                    axes[6].imshow(np.transpose(x_scaled[lookahead-1:, :]), aspect='auto', vmin=0.0, vmax=1.0)
                else:
                    axes[0].imshow(np.transpose(y), aspect='auto', vmin=0.0, vmax=1.0)
                    axes[6].imshow(np.transpose(x_scaled), aspect='auto', vmin=0.0, vmax=1.0)

                axes[5].imshow(np.transpose(activations_0), aspect='auto', vmin=0.0, vmax=1.0)
                axes[4].imshow(np.transpose(activations_1), aspect='auto', vmin=0.0, vmax=1.0)
                axes[3].imshow(np.transpose(activations_2), aspect='auto', vmin=0.0, vmax=1.0)
                axes[2].imshow(np.transpose(activations_3), aspect='auto', vmin=0.0, vmax=1.0)
                axes[1].imshow(np.transpose(predictions), aspect='auto', vmin=0.0, vmax=1.0)

            model_str = str_model[model_n].replace('C:\\Users\\Harry\\Documents\\DNN_Project\\models\\', '').replace('_20Epochs_5Lookahead_11.h5', '')
            print('Model %s: loss = %.2f  accuracy = %.2f%%' % (model_str, loss, accuracy*100))
            print('Test Acc = %.2f%%' % (accuracy *100))
            print('False activation rate = %.2f%% of %.2f%% of frames' % (FAR * 100, nonspeech_frames * 100))
            print('Miss rate = %.2f%% of %.2f%% of frames' % (MR * 100, speech_frames * 100))
            entry = pd.DataFrame({'Noise': [noise],
                                  'Model': [model_str],
                                  'Loss': [loss],
                                  'Accuracy': [accuracy],
                                  'FAR': [FAR],
                                  'MR': [MR]})
            output_df = output_df.append(entry, ignore_index=True)

            output = np.empty(predictions.shape)
            for i in range(predictions.shape[0]):
                if predictions[i] < 0.5:
                    output[i] = 0
                else:
                    output[i] = 1

            np.savetxt('C:\\Users\\Harry\\Documents\\DNN_Project\\Targets_%ddB.csv' % snr, output, delimiter=',')

#output_df.to_csv('C:\\Users\\Harry\\Documents\\DNN_Project\\PerformanceData_run%d.csv' % run)
if plot:
    plt.show()