"""

Model: Long short-term memory (LSTM) with dense (i.e. fully connected) layers
Mehtod: Truncated Backpropagation Through Time (TBPTT)

Architecture: Recurrent Neural Network
Dataset: Monthly sunspots

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

import json, yaml

import argparse

import os

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

def series_to_classification(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

n_series = len(sunspots)

n_train = int(n_series * 2/3) # number of training examples/samples
n_test = n_series - n_train  # number of test examples/samples

train_x = np.arange(n_train)
test_x = np.arange(n_train, n_series) + 1

train_y = sunspots[:n_train]
test_y = sunspots[n_train:]

#%% 
# PREPROCESSING STEP
scaling_factor = args.scaling_factor
translation = args.translation

# Set up the model and the methods





