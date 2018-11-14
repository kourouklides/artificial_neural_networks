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
from keras.layers import Input, Dense, LSTM
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

import json, yaml

import argparse

import os

import matplotlib.pyplot as plt

initial_dir = os.getcwd()
os.chdir( '../../../../../' )
from artificial_neural_networks.code.utils.download_mnist import download_monthly_sunspots

#%% 
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
parser.add_argument('--look_back', type = int, default = 3)
parser.add_argument('--scaling_factor', type = float, default = (1/780) )
parser.add_argument('--translation', type = float, default = 0)
parser.add_argument('--layer_size', type = int, default = 4)
parser.add_argument('--n_epochs', type = int, default = 13)
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

arguments = parser.parse_args()

#%% 
def lstm_dense_sunsposts(args):
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
    
    sunspots_path = download_monthly_sunspots()
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
    
    x = Input(shape = (n_in,1)) #input layer
    h = x
    
    h = LSTM(units = args.layer_size)(h) # hidden layer
    
    out = Dense(units = n_out, activation = None)(h) # output layer
    
    model = Model(inputs = x, outputs = out)
    
    if (args.verbose > 0):
        model.summary()
    
    def root_mean_squared_error(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
    
    loss_function = root_mean_squared_error
    
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
    
    model_history = model.fit(x = train_x_, y = train_y_, \
                              validation_data = (test_x_, test_y_), \
                              batch_size = args.batch_size, \
                              epochs = args.n_epochs, \
                              verbose = args.verbose, \
                              callbacks = callbacks)
    
    #%% 
    # TESTING PHASE
    
    # Predict preprocessed values
    train_y_pred_ = model.predict(train_x_)[:,0]
    test_y_pred_ = model.predict(test_x_)[:,0]
    
    # Remove preprocessing
    train_y_pred = affine_transformation(train_y_pred_, scaling_factor, translation, \
                                         inverse = True)
    test_y_pred = affine_transformation(test_y_pred_, scaling_factor, translation, \
                                        inverse = True)
    
    train_rmse = sqrt(mean_squared_error(train_y, train_y_pred))
    train_mae = mean_absolute_error(train_y, train_y_pred)
    train_r2 = r2_score(train_y, train_y_pred)
    
    test_rmse = sqrt(mean_squared_error(test_y, test_y_pred))
    test_mae = mean_absolute_error(test_y, test_y_pred)
    test_r2 = r2_score(test_y, test_y_pred)
    
    if (args.verbose > 0):
        print('Train RMSE: %.4f ' % (train_rmse))
        print('Train MAE: %.4f ' % (train_mae))
        print('Train (1 - R_squared): %.4f ' % (1.0 - train_r2))
        print('Train R_squared: %.4f ' % (train_r2))
        print('')
        print('Test RMSE: %.4f ' % (test_rmse))
        print('Test MAE: %.4f ' % (test_mae))
        print('Test (1 - R_squared): %.4f ' % (1.0 - test_r2))
        print('Test R_squared: %.4f ' % (test_r2))
    
    #%% 
    # Data Visualization
    
    if (args.plot):
        plt.plot(train_y)
        plt.plot(train_y_pred)
        plt.title('Time Series of the training set')
        plt.show()
        
        plt.plot(test_y)
        plt.plot(test_y_pred)
        plt.title('Time Series of the test set')
        plt.show()
        
        train_errors = train_y - train_y_pred
        plt.hist(train_errors, bins='auto')
        plt.title('Histogram of training errors')
        plt.show()
        
        test_errors = test_y - test_y_pred
        plt.hist(test_errors, bins='auto')
        plt.title('Histogram of test errors')
        plt.show()
        
        plt.scatter(x = train_y, y = train_y_pred, edgecolors=(0, 0, 0))
        plt.plot([train_y.min(), train_y.max()], [train_y.min(), train_y.max()], 'k--', lw=4)
        plt.title('Predicted vs Actual for training set')
        plt.show()
        
        plt.scatter(x = test_y, y = test_y_pred, edgecolors=(0, 0, 0))
        plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=4)
        plt.title('Predicted vs Actual for test set')
        plt.show()
        
        plt.scatter(x = train_y_pred, y = train_errors, edgecolors=(0, 0, 0))
        plt.plot([train_y.min(), train_y.max()], [0, 0], 'k--', lw=4)
        plt.title('Residuals vs Predicted for training set')
        plt.show()
        
        plt.scatter(x = test_y_pred, y = test_errors, edgecolors=(0, 0, 0))
        plt.plot([test_y.min(), test_y.max()], [0, 0], 'k--', lw=4)
        plt.title('Residuals vs Predicted for test set')
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
    
    return model

#%% 
if __name__ == '__main__':
    model = lstm_dense_sunsposts(args = arguments)
