"""

Model: Persistence Model
Mehtod: Naive (i.e. copying the previous value)

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

from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM

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
parser.add_argument('--scaling_factor', type = float, default = (1/255) )
parser.add_argument('--translation', type = float, default = 0)

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
		a = dataset[i:(i + look_back)]
		data_X.append(a)
		data_Y.append(dataset[i + look_back])
        
	return np.array(data_X), np.array(data_Y)

n_series = len(sunspots)

split_ratio = 2/3 # between zero and one
n_split = int(n_series * split_ratio)

train = sunspots[:n_split]
test = sunspots[n_split:]

look_back = 1
train_x, train_y = series_to_supervised(train,look_back)
test_x, test_y = series_to_supervised(test,look_back)

#%% 
# PREPROCESSING STEP
scaling_factor = args.scaling_factor
translation = args.translation

# Set up the model and the methods

n_train = train_x.shape[0] # number of training examples/samples
n_test = test_x.shape[0] # number of test examples/samples

n_in = train_x.shape[1] # number of features / dimensions
n_out = 1 # number of classes/labels

# Reshape training and test sets
train_x = train_x.reshape(n_train, n_in, 1)
test_x = test_x.reshape(n_test, n_in, 1)

# Apply preprocessing
train_x = scaling_factor * (train_x - translation)
test_x = scaling_factor * (test_x - translation)

#%% 
# TRAINING PHASE

model_history = model.fit(x = train_x, y = train_y, \
                          epochs=10, batch_size=1, verbose=1)

#%% 
# TESTING PHASE

train_y_pred = model.predict(train_x)
test_y_pred = model.predict(test_x)

train_rmse = sqrt(mean_squared_error(train_y, train_y_pred))
train_mae = mean_absolute_error(train_y, train_y_pred)
train_r2 = r2_score(train_y, train_y_pred)
print('Train RMSE: %.2f ' % (train_rmse))
print('Train MAE: %.2f ' % (train_mae))
print('Train R-squared: %.2f ' % (train_r2))

print('')

test_rmse = sqrt(mean_squared_error(test_y, test_y_pred))
test_mae = mean_absolute_error(test_y, test_y_pred)
test_r2 = r2_score(test_y, test_y_pred)
print('Test RMSE: %.2f ' % (test_rmse))
print('Test MAE: %.2f ' % (test_mae))
print('Test R-squared: %.2f ' % (test_r2))

#%% 
# Data Visualization

plt.plot(train_y)
plt.plot(train_y_pred)
plt.show()

plt.plot(test_y)
plt.plot(test_y_pred)
plt.show()

