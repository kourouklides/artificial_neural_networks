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

mnist_path = r'../../../../datasets/mnist.npz'
mnist = np.load(mnist_path)
train_x = mnist['x_train'].astype(np.float32)
train_y = mnist['y_train'].astype(np.int32)
test_x = mnist['x_test'].astype(np.float32)
test_y = mnist['y_test'].astype(np.int32)
mnist.close()














