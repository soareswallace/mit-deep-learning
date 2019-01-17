#tensor flow libraries
import tensorflow as tf
from tensorflow import keras
#numpy libraries
import numpy as np
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cv2
import IPython

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: 
            print('')
        print ('.', end='')

def build_model(train_features):
    model = keras.Sequential([
        keras.layers.Dense(20, activation=tf.nn.relu, input_shape=[len(train_features[0])]),
        keras.layers.Dense(100),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(), loss='mse', metrics=['mae', 'mse'])

    return model

def plot_history(history):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Square Error [Thousand Dollars$^2$]")
    plt.plot(history['epoch'], history['mean_squared_error'], label='Train Error')
    plt.plot(history['epoch'], history['val_mean_squared_error'], label='Val Error')
    plt.legend()
    plt.ylim([0, 50])

    plt.show()



(train_features, train_labels), (test_features, test_labels) = keras.datasets.boston_housing.load_data()
#calcula a media
train_mean = np.mean(train_features, axis=0)
#calcula o desvio padrão
train_std = np.std(train_features, axis=0)
#normalizacao dos dados
train_features_norm = (train_features-train_mean)/train_std

model = build_model(train_features_norm)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
history = model.fit(train_features_norm, train_labels, batch_size=32, epochs=10000, verbose=True, validation_split=0.1, callbacks=[early_stop, PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

rmse_final = np.sqrt(float(hist['val_mean_squared_error'].tail(1)))
print()
print('Final Root Mean Square Error on validation set: {}'.format(round(rmse_final, 3)))

test_mean = np.mean(test_features, axis=0)
test_std = np.std(test_features, axis=0)

test_features_norm = (test_features - train_mean) / train_std
mse, _, _ = model.evaluate(test_features_norm, test_labels)
predict = model.predict(test_features_norm)

rmse = np.sqrt(mse)
print('Root Mean Square Error on test set: {}'.format(rmse, 3))

plt.xlabel("lstat")
plt.ylabel("value")
plt.scatter(train_features[:, -1], train_labels)

plot_history(hist)

print (test_labels)
print (predict)