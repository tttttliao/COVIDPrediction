import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import datetime
from math import sqrt
from numpy import split
from numpy import array
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras import optimizers 
from keras.layers import Input
from keras import layers
from keras.models import Model

def plot_loss(r):
    plt.plot(r.history['loss'])
    plt.plot(r.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

def build_and_fit(train_x, train_x_demo, train_y_1, train_y_2, n_input, n_out, epochs_num, batch_size_num):
    
    #set parameters
    epochs, batch_size = epochs_num, batch_size_num
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y_1.shape[1]
    
    #reshape output into [samples, timesteps, features]
    train_y_1 = train_y_1.reshape((train_y_1.shape[0], train_y_1.shape[1], 1))
    train_y_2 = train_y_2.reshape((train_y_2.shape[0], train_y_2.shape[1], 1))
    
    #---define model---#
    #time_series
    time_series_input = Input(shape=(n_timesteps, n_features))
    lstm_main = layers.TimeDistributed(Dense(2))(time_series_input)
    
    #one hot matrix of states reshaped => shape similarly as time_series_input
    state_input = Input(shape=(n_out, 50))
    state_dense = layers.Dense(10)(state_input)
    state_dropout = layers.Dropout(0.2)(state_dense)
    
    #for confirmed
    h1_c, h2_c, c_c = layers.LSTM(32, inner_init='orthogonal', return_sequences=True, return_state = True)(lstm_main)
    decoder_c = layers.RepeatVector(n_out)(h2_c)
    decoder_c = layers.LSTM(32, dropout = 0.5, recurrent_dropout=0.2,
                         return_sequences=True, return_state = False)(decoder_c, [h2_c, c_c])
    attention_c = layers.dot([decoder_c, h1_c], axes = [2,2])
    attention_c = layers.Activation('softmax')(attention_c)
    context_c = layers.dot([attention_c, h1_c], axes = [2,1])
    decoder_and_context_c = layers.Concatenate(axis=2)([context_c, decoder_c])
    merge_c = layers.Concatenate(axis=2)([decoder_and_context_c, state_dropout])
    dense_c = layers.TimeDistributed(Dense(10))(merge_c)
    dropout_c = layers.Dropout(0.5)(dense_c)
    output_c = layers.TimeDistributed(Dense(1))(merge_c)
    confirmed = layers.LeakyReLU(alpha=0.1, name = 'confirmed')(output_c)
    
    
    #for deaths
    h1_d, h2_d, c_d = layers.LSTM(32, inner_init='orthogonal', return_sequences=True, return_state = True)(lstm_main)
    decoder_d = layers.RepeatVector(n_out)(h2_d)
    decoder_d = layers.LSTM(32, dropout = 0.5, recurrent_dropout=0.2,
                         return_sequences=True, return_state = False)(decoder_d, [h2_d, c_d])
    attention_d = layers.dot([decoder_c, h1_c], axes = [2,2])
    attention_d = layers.Activation('softmax')(attention_d)
    context_d = layers.dot([attention_d, h1_d], axes = [2,1])
    decoder_and_context_d = layers.Concatenate(axis=2)([context_d, decoder_d])
    merge_d = layers.Concatenate(axis=2)([decoder_and_context_d, state_dropout])
    dense_d = layers.TimeDistributed(Dense(10))(merge_d)
    dropout_d = layers.Dropout(0.5)(dense_d)
    output_d = layers.TimeDistributed(Dense(1))(merge_d)
    deaths = layers.LeakyReLU(alpha=0.1, name = 'deaths')(output_d)
    
    #put together and compile model
    model = Model([time_series_input,state_input], [confirmed,deaths])
    #opt = optimizers.Adam(lr = 0.001, clipnorm = 1.)
    model.compile(loss='mse', optimizer='Adam')
    
    #check shape
    print(train_x.shape)
    print(train_x_demo.shape)
    print(train_y_1.shape)
    print(train_y_2.shape)
    
    #fit model
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    record = model.fit([train_x,train_x_demo], [train_y_1, train_y_2], validation_split=0.2, callbacks = [es],
                       epochs=epochs, batch_size=batch_size, verbose=0)
    #plot loss
    plot_loss(record)
    
    return model

def forecast(model, history, history_demo, n_in, n_out):
    # flatten data
    data = np.array(history)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_in:, :]
    # reshape into [1, n_input, n]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    # reshape into [1, n_input, num_of_features]
    input_x_demo = history_demo.reshape(1,50)
    input_x_demo = np.repeat(input_x_demo, n_out, axis=0)
    input_x_demo = input_x_demo.reshape((1, n_out, 50))
    # forecast 
    yhat = model.predict([input_x, input_x_demo], verbose=0)
    yhat1 = yhat[0]
    yhat2 = yhat[1]
    return yhat1, yhat2

def run_model(train_x, train_x_demo, train_y_1, train_y_2, test, train_ls, demo_ls,
                   n_input, n_out, epochs = 80, batch_size = 128):
    # build and fit model
    model = build_and_fit(train_x, train_x_demo, train_y_1, train_y_2, n_input, n_out, epochs, batch_size)
    predictions_1 = [[] for i in range(50)]
    predictions_2 = [[] for i in range(50)]
    # forcase in n_input days at a time, total 28/n_input iterations
    for i in range(int(28/n_out)):
        # forcast each state
        for j in range(50):
            yhat_sequence_1, yhat_sequence_2 = forecast(model, train_ls[j], demo_ls[j], n_input, n_out)
            predictions_1[j].append(yhat_sequence_1)
            predictions_2[j].append(yhat_sequence_2)
            yhat = np.concatenate((yhat_sequence_1.reshape(-1,1), yhat_sequence_2.reshape(-1,1)), axis = 1)
            yhat = yhat.reshape(-1,n_out,2) #2 is the number of features; should use 7 to replace n_out
            #for s in range(int(n_input/7)):
             #   train_ls[j] = np.vstack((train_ls[j], yhat[0][s*7:(s+1)*7].reshape(-1, 7, 2))) #7 is base length
            # add newly predicted data to train data for next round of predictions
            train_ls[j] = np.vstack((train_ls[j], yhat))
    return predictions_1, predictions_2, model