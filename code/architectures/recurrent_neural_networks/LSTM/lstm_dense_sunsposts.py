"""

Model: Long short-term memory (LSTM) with dense (i.e. fully connected) layers
Mehtod: Truncated Backpropagation Through Time (TBPTT)
Architecture: Recurrent Neural Network

Dataset: Monthly sunspots
Task: Time Series Forecasting

    Author: Ioannis Kourouklides, www.kourouklides.com
    License: https://github.com/kourouklides/artificial_neural_networks/blob/master/LICENSE

"""
#%% 
# Python configurations

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random as rn

from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from keras.layers import Input, Dense, LSTM
from keras.models import Model
from keras.callbacks import ModelCheckpoint

import json, yaml

import argparse

import os

import matplotlib.pyplot as plt

# SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--verbose', type = int, default = 1)
parser.add_argument('--reproducible', type = bool, default = True)
parser.add_argument('--seed', type = int, default = 0)

# Settings for preprocessing and hyperparameters
parser.add_argument('--scaling_factor', type = float, default = (1/355) )
parser.add_argument('--translation', type = float, default = 123)
parser.add_argument('--n_epochs', type = int, default = 8)
parser.add_argument('--batch_size', type = int, default = 1)
parser.add_argument('--optimizer', type = str, default = 'adam')
parser.add_argument('--lrearning_rate', type = float, default = 1e0)
parser.add_argument('--epsilon', type = float, default = None)

# Settings for saving the model
parser.add_argument('--save_architecture', type = bool, default = True)
parser.add_argument('--save_last_weights', type = bool, default = True)
parser.add_argument('--save_last_model', type = bool, default = True)
parser.add_argument('--save_models', type = bool, default = False)
parser.add_argument('--save_weights_only', type = bool, default = False)
parser.add_argument('--save_best_only', type = bool, default = False)
args = parser.parse_args()

if (args.verbose > 0):
    print(args)

# For reproducibility
if (args.reproducible):
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(args.seed)
    rn.seed(args.seed)
    tf.set_random_seed(args.seed)
    
#%% 
# Load the Montly sunspots dataset

sunspots_path = r'../../../../datasets/monthly-sunspots.csv'
sunspots = np.genfromtxt(fname=sunspots_path, dtype = np.float32,  \
                        delimiter = ",", skip_header = 1, usecols = 1)

#%% 
# Train-Test split

def series_to_supervised(dataset, look_back = 1):
    # Prepare the dataset (Time Series) to be used for Supervised Learning
    
    data_X, data_Y = [], []
    
    for i in range(len(dataset) - look_back):
        sliding_window = i + look_back
        data_X.append(dataset[i:sliding_window])
        data_Y.append(dataset[sliding_window])
        
    return np.array(data_X), np.array(data_Y)

n_series = len(sunspots)

split_ratio = 2/3 # between zero and one
n_split = int(n_series * split_ratio)

look_back = 3

train = sunspots[:n_split + look_back]
test = sunspots[n_split:]

train_x, train_y = series_to_supervised(train,look_back)
test_x, test_y = series_to_supervised(test,look_back)

#%% 
# PREPROCESSING STEP

scaling_factor = args.scaling_factor
translation = args.translation

n_train = train_x.shape[0] # number of training examples/samples
n_test = test_x.shape[0] # number of test examples/samples

n_in = train_x.shape[1] # number of features / dimensions
n_out = 1 # number of classes/labels

# Reshape training and test sets
train_x = train_x.reshape(n_train, n_in, 1)
test_x = test_x.reshape(n_test, n_in, 1)

def affine_transformation(data_in, scaling, translation, inverse = False):
    # Apply affine trasnforamtion to the data
    
    if (inverse):
        #Inverse Transformation
        data_out = (data_in / scaling) + translation
    else:
        # Direct Transformation
        data_out = scaling * (data_in - translation)
    return data_out

# Apply preprocessing
train_x_ = affine_transformation(train_x, scaling_factor, translation)
train_y_ = affine_transformation(train_y, scaling_factor, translation)
test_x_ = affine_transformation(test_x, scaling_factor, translation)
test_y_ = affine_transformation(test_y, scaling_factor, translation)

#%% 
# Model hyperparameters

# ANN Architecture

x = Input(shape = (n_in, 1)) #input layer
h = x

h = LSTM(units = 4)(h)

out = Dense(units = n_out, activation = None)(h) # output layer

model = Model(inputs = x, outputs = out)

if (args.verbose > 0):
    model.summary()

loss_function = 'mean_squared_error'

metrics = ['mean_absolute_error', 'mean_absolute_percentage_error']

optimizer = 'adam'

model.compile(optimizer = optimizer, \
              loss = loss_function, \
              metrics = metrics)

#%% 
# Save trained models for every epoch

models_path = r'../../../trained_models/'
model_name = 'lstm_dense_sunspots'
weights_path = models_path + model_name + '_weights'
model_path = models_path + model_name + '_model'
file_suffix = '_{epoch:04d}_{val_acc:.4f}_{val_loss:.4f}'

if (args.save_weights_only):
    file_path = weights_path
else:
    file_path = model_path

file_path += file_suffix

# monitor = 'val_loss'
monitor = 'val_acc'

if (args.save_models):
    checkpoint = ModelCheckpoint(file_path + '.h5', \
                                 monitor = monitor, \
                                 verbose = args.verbose, \
                                 save_best_only = args.save_best_only, \
                                 mode='auto', \
                                 save_weights_only = args.save_weights_only)
    callbacks = [checkpoint]
else:
    callbacks = []

#%% 
# TRAINING PHASE

model_history = model.fit(x = train_x_, y = train_y_, \
                          validation_data = (test_x_, test_y_), \
                          batch_size = args.batch_size, \
                          epochs = args.n_epochs, \
                          verbose = args.verbose, \
                          callbacks = callbacks)

#%% 
# TESTING PHASE

# Predict preprocessed values
train_y_pred_ = model.predict(train_x_)
test_y_pred_ = model.predict(test_x_)

# Remove preprocessing
train_y_pred = affine_transformation(train_y_pred_, scaling_factor, translation, inverse = True)
test_y_pred = affine_transformation(test_y_pred_, scaling_factor, translation, inverse = True)

train_rmse = sqrt(mean_squared_error(train_y, train_y_pred))
train_mae = mean_absolute_error(train_y, train_y_pred)
train_r2 = r2_score(train_y, train_y_pred)
print('Train RMSE: %.4f ' % (train_rmse))
print('Train MAE: %.4f ' % (train_mae))
print('Train (1- R_squared): %.4f ' % (1.0 - train_r2))
print('Train R_squared: %.4f ' % (train_r2))


print('')

test_rmse = sqrt(mean_squared_error(test_y, test_y_pred))
test_mae = mean_absolute_error(test_y, test_y_pred)
test_r2 = r2_score(test_y, test_y_pred)
print('Test RMSE: %.4f ' % (test_rmse))
print('Test MAE: %.4f ' % (test_mae))
print('Test (1- R_squared): %.4f ' % (1.0 - test_r2))
print('Test R_squared: %.4f ' % (test_r2))

#%% 
# Data Visualization

plt.plot(train_y)
plt.plot(train_y_pred)
plt.show()

plt.plot(test_y)
plt.plot(test_y_pred)
plt.show()

