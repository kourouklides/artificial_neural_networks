"""

Model: Long short-term memory (LSTM) with dense (i.e. fully connected) layers
Mehtod: Truncated Backpropagation Through Time (TBPTT)
Architecture: Recurrent Neural Network

Dataset: Monthly sunspots
Task: Time Series Forecasting (Univariate Regression)

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

from keras import optimizers
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint

from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import json, yaml

import argparse

import os

import matplotlib.pyplot as plt

def none_or_int(value):
    if value == 'None':
        return None
    else:
        return int(value)
    
def none_or_float(value):
    if value == 'None':
        return None
    else:
        return float(value)

# SETTINGS
parser = argparse.ArgumentParser()

# General settings
parser.add_argument('--verbose', type = int, default = 1)
parser.add_argument('--reproducible', type = bool, default = True)
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--plot', type = bool, default = True)

# Settings for preprocessing and hyperparameters
parser.add_argument('--look_back', type = int, default = 10)
parser.add_argument('--scaling_factor', type = float, default = (1/780) )
parser.add_argument('--translation', type = float, default = 0)
parser.add_argument('--same_size', type = bool, default = False)
parser.add_argument('--n_layers', type = int, default = 2)
parser.add_argument('--layer_size', type = int, default = 128)
parser.add_argument('--explicit_layer_sizes', nargs='*', type=int, default = [128, 128])
parser.add_argument('--n_epochs', type = int, default = 7)
parser.add_argument('--batch_size', type = none_or_int, default = 1)
parser.add_argument('--optimizer', type = str, default = 'Adam')
parser.add_argument('--lrearning_rate', type = float, default = 1e-3)
parser.add_argument('--epsilon', type = none_or_float, default = None)

# Settings for saving the model
parser.add_argument('--save_architecture', type = bool, default = True)
parser.add_argument('--save_last_weights', type = bool, default = True)
parser.add_argument('--save_last_model', type = bool, default = True)
parser.add_argument('--save_models', type = bool, default = False)
parser.add_argument('--save_weights_only', type = bool, default = False)
parser.add_argument('--save_best', type = bool, default = False)

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
# Load the Monthly sunspots dataset

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

look_back = args.look_back

train = sunspots[:n_split + look_back]
test = sunspots[n_split:]

train_x, train_y = series_to_supervised(train,look_back)
test_x, test_y = series_to_supervised(test,look_back)

# Apply diferencing for Seasonality Adjustment to make the it stationary
d_train = train[1:] - train[:-1]
d_test = test[1:] - test[:-1]

train_first = train[look_back]
test_first = test[look_back]

d_train_x, d_train_y = series_to_supervised(d_train, look_back)
d_test_x, d_test_y = series_to_supervised(d_test, look_back)

#%% 
# PREPROCESSING STEP

scaling_factor = args.scaling_factor
translation = args.translation

n_train = d_train_x.shape[0] # number of training examples/samples
n_test = d_test_x.shape[0] # number of test examples/samples

n_in = d_train_x.shape[1] # number of features / dimensions
n_out = 1 # number of classes/labels

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
d_train_x_ = affine_transformation(d_train_x, scaling_factor, translation)
d_train_y_ = affine_transformation(d_train_y, scaling_factor, translation)
d_test_x_ = affine_transformation(d_test_x, scaling_factor, translation)
d_test_y_ = affine_transformation(d_test_y, scaling_factor, translation)

#%% 
# Model hyperparameters

N = []
N.append(n_in) #input layer
if (args.same_size):
    n_layers = args.n_layers
    for i in range(n_layers):
        N.append(args.layer_size) # hidden layer i
else:
    n_layers = len(args.explicit_layer_sizes)
    for i in range(n_layers):
        N.append(args.explicit_layer_sizes[i]) # hidden layer i
N.append(n_out) # output layer

# ANN Architecture

L = len(N) - 1

x = Input(shape = (n_in,)) #input layer
h = x

for i in range(1,L):
    h = Dense(units = N[i], activation = 'relu')(h) # hidden layer i

out = Dense(units = n_out, activation = None)(h) # output layer

model = Model(inputs = x, outputs = out)

if (args.verbose > 0):
    model.summary()

loss_function = 'mean_squared_error'

metrics = ['mean_absolute_error', 'mean_absolute_percentage_error']

lr = args.lrearning_rate
epsilon = args.epsilon
optimizer_selection = {'Adadelta' : optimizers.Adadelta( \
                               lr=lr, rho=0.95, epsilon=epsilon, decay=0.0), \
                       'Adagrad' : optimizers.Adagrad( \
                               lr=lr, epsilon=epsilon, decay=0.0), \
                       'Adam' : optimizers.Adam( \
                               lr=lr, beta_1=0.9, beta_2=0.999, \
                               epsilon=epsilon, decay=0.0, amsgrad=False), \
                       'Adamax' : optimizers.Adamax( \
                               lr=lr, beta_1=0.9, beta_2=0.999, \
                               epsilon=epsilon, decay=0.0), \
                       'Nadam' : optimizers.Nadam( \
                               lr=lr, beta_1=0.9, beta_2=0.999, \
                               epsilon=epsilon, schedule_decay=0.004), \
                       'RMSprop' : optimizers.RMSprop( \
                               lr=lr, rho=0.9, epsilon=epsilon, decay=0.0), \
                       'SGD' : optimizers.SGD( \
                               lr=lr, momentum=0.0, decay=0.0, nesterov=False)}

optimizer = optimizer_selection[args.optimizer]

model.compile(optimizer = optimizer, \
              loss = loss_function, \
              metrics = metrics)

#%% 
# Save trained models for every epoch

models_path = r'../../../../trained_models/'
model_name = 'sunspots_lstm_dense'
weights_path = models_path + model_name + '_weights'
model_path = models_path + model_name + '_model'
file_suffix = '_{epoch:04d}_{val_loss:.4f}_{val_mean_absolute_error:.4f}'

if (args.save_weights_only):
    file_path = weights_path
else:
    file_path = model_path

file_path += file_suffix

monitor = 'val_loss'

if (args.save_models):
    checkpoint = ModelCheckpoint(file_path + '.h5', \
                                 monitor = monitor, \
                                 verbose = args.verbose, \
                                 save_best_only = args.save_best, \
                                 mode='auto', \
                                 save_weights_only = args.save_weights_only)
    callbacks = [checkpoint]
else:
    callbacks = []

#%% 
# TRAINING PHASE

model_history = model.fit(x = d_train_x_, y = d_train_y_, \
                          validation_data = (d_test_x_, d_test_y_), \
                          batch_size = args.batch_size, \
                          epochs = args.n_epochs, \
                          verbose = args.verbose, \
                          callbacks = callbacks)

#%% 
# TESTING PHASE

# Predict values (with preprocessing and differencing)
d_train_y_pred_ = model.predict(d_train_x_)
d_test_y_pred_ = model.predict(d_test_x_)

# Remove preprocessing
d_train_y_pred = affine_transformation(d_train_y_pred_, scaling_factor, translation, \
                                     inverse = True)
d_test_y_pred = affine_transformation(d_test_y_pred_, scaling_factor, translation, \
                                    inverse = True)

# Remove differencing
train_y_pred = np.cumsum(np.insert(d_train_y_pred, 0, train_first))
test_y_pred = np.cumsum(np.insert(d_test_y_pred, 0, test_first))


"""

train_y_pred, test_y_pred = d_train_y_pred, d_test_y_pred
train_y, test_y = d_train_y, d_test_y

"""

train_rmse = sqrt(mean_squared_error(train_y, train_y_pred))
train_mae = mean_absolute_error(train_y, train_y_pred)
train_r2 = r2_score(train_y, train_y_pred)

test_rmse = sqrt(mean_squared_error(test_y, test_y_pred))
test_mae = mean_absolute_error(test_y, test_y_pred)
test_r2 = r2_score(test_y, test_y_pred)

if (args.verbose > 0):
    print('Train RMSE: %.4f ' % (train_rmse))
    print('Train MAE: %.4f ' % (train_mae))
    print('Train (1- R_squared): %.4f ' % (1.0 - train_r2))
    print('Train R_squared: %.4f ' % (train_r2))
    print('')
    print('Test RMSE: %.4f ' % (test_rmse))
    print('Test MAE: %.4f ' % (test_mae))
    print('Test (1- R_squared): %.4f ' % (1.0 - test_r2))
    print('Test R_squared: %.4f ' % (test_r2))

#%% 
# Data Visualization

if (args.plot):
    plt.plot(train_y)
    plt.plot(train_y_pred)
    plt.show()
    
    plt.plot(test_y)
    plt.plot(test_y_pred)
    plt.show()

#%% 
# Save the architecture and the lastly trained model

architecture_path = models_path + model_name + '_architecture'

last_suffix = file_suffix.format(epoch = args.n_epochs, \
                                 val_loss = test_rmse, \
                                 val_mean_absolute_error = test_mae)

if (args.save_architecture):
    # Save only the archtitecture (as a JSON file)
    json_string = model.to_json()
    json.dump(json.loads(json_string), open(architecture_path + '.json', "w"))
    
    # Save only the archtitecture (as a YAML file)
    yaml_string = model.to_yaml()
    yaml.dump(yaml.load(yaml_string), open(architecture_path + '.yml', "w"))

# Save only the weights (as an HDF5 file)
if (args.save_last_weights):
    model.save_weights(weights_path + last_suffix + '.h5')

# Save the whole model (as an HDF5 file)
if (args.save_last_model):
    model.save(model_path + last_suffix + '.h5')

