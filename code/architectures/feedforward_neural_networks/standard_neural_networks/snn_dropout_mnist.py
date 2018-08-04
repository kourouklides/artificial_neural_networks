"""

Model: Standard Neural Network (SNN) with dropout layers
Mehtod: Backpropagation

Architecture: Feedforward Neural Network
Dataset: MNIST

    Author: Ioannis Kourouklides, www.kourouklides.com

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
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

import json, yaml

import argparse

import os

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--verbose', type = int, default = )
parser.add_argument('--reproducible', type = bool, default = True)
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--n_layers', type = int, default = 2)
parser.add_argument('--layer_size', type = int, default = 512)
parser.add_argument('--n_epochs', type = int, default = 20)
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--save_architecture', type = bool, default = True)
parser.add_argument('--save_last_weights', type = bool, default = True)
parser.add_argument('--save_last_model', type = bool, default = True)
parser.add_argument('--save_models', type = bool, default = False)
parser.add_argument('--save_weights_only', type = bool, default = False)
parser.add_argument('--save_best_only', type = bool, default = False)
parser.add_argument('--dropout_rate', type = int, default = 0.2)
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
# Load MNIST data

mnist_path = '../../../../data/mnist.npz'
mnist = np.load(mnist_path)
train_x = mnist['x_train'].astype(np.float32)
train_y = mnist['y_train'].astype(np.int32)
test_x = mnist['x_test'].astype(np.float32)
test_y = mnist['y_test'].astype(np.int32)
mnist.close()

#%% 
# Set up the model and the methods

img_width = train_x.shape[1]
img_length = train_x.shape[2]

n_train = train_x.shape[0] # number of training examples/samples
n_test = test_x.shape[0] # number of test examples/samples
n_in = img_width * img_length # number of features / dimensions
n_out = np.unique(train_y).shape[0] # number of classes/labels

# PREPROCESSING STEP
scaling_factor = (1/255) 

# Reshape training and test sets
train_x = scaling_factor * train_x.reshape(n_train, n_in)
test_x = scaling_factor * test_x.reshape(n_test, n_in)

one_hot = False

# Convert class vectors to binary class matrices (i.e. One hot encoding)
if (one_hot):
    train_y = to_categorical(train_y, n_out)
    test_y = to_categorical(test_y, n_out)

# Model hyperparameters
N = []
N.append(n_in) #input layer
for i in range(args.n_layers):
    N.append(args.layer_size) # hidden layer i
N.append(n_out) # output layer
# N = [n_in, 64, 128, 64, n_out]
lrearning_rate = 1e-3
epsilon = None
optimizer = optimizers.RMSprop(lr=lrearning_rate,epsilon=epsilon)
# optimizer = optimizers.Adam(lr=lrearning_rate,epsilon=epsilon)

# ANN Architecture
L = len(N) - 1
x = Input(shape = (n_in,)) #input layer
h = x
for i in range(1,L):
    h = Dense(N[i], activation = 'relu')(h) # hidden layer i
    h = Dropout(args.dropout_rate)(h)
out = Dense(n_out, activation = 'softmax')(h) # output layer

model = Model(inputs = x, outputs = out)

if (args.verbose > 0):
    model.summary()

if (one_hot):
    loss_function = 'categorical_crossentropy'
else:
    loss_function = 'sparse_categorical_crossentropy'

metrics=['accuracy']

model.compile(optimizer = optimizer, \
              loss = loss_function, \
              metrics = metrics)

#%% 
# Save trained models for every epoch

models_path = '../../../../trained_models/'
model_name = 'mnist_snn_dropout'
weights_path = models_path + model_name + '_weights'
model_path = models_path + model_name + '_model'

if (args.save_weights_only):
    file_path = weights_path
else:
    file_path = model_path
file_path += '_{epoch:04d}_{val_acc:.4f}_{val_loss:.4f}'

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

model_history = model.fit(x = train_x, y = train_y,
                          validation_data = (test_x, test_y), \
                          batch_size = args.batch_size, \
                          epochs = args.n_epochs, \
                          verbose = args.verbose, \
                          callbacks = callbacks)

#%% 
# VALIDATION PHASE

score = model.evaluate(x = test_x, y = test_y, verbose = args.verbose)
score_dict = {'val_loss' : score[0], 'val_acc' : score[1]}

if (args.verbose > 0):
    print('Test loss:', score_dict['val_loss'])
    print('Test accuracy:', score_dict['val_acc'])

#%% 
# Save the architecture and the lastly trained model

architecture_path = models_path + model_name + '_architecture'

last = '_{0:04d}_{1:.4f}_{2:.4f}'.format(args.n_epochs, \
                                         score_dict['val_acc'], \
                                         score_dict['val_loss'])

if (args.save_architecture):
    # Save only the archtitecture (as a JSON file)
    json_string = model.to_json()
    json.dump(json.loads(json_string), open(architecture_path + '.json', "w"))
    
    # Save only the archtitecture (as a YAML file)
    yaml_string = model.to_yaml()
    yaml.dump(yaml.load(yaml_string), open(architecture_path + '.yml', "w"))

# Save only the weights (as an HDF5 file)
if (args.save_last_weights):
    model.save_weights(weights_path + last + '.h5')

# Save the whole model (as an HDF5 file)
if (args.save_last_model):
    model.save(model_path + last + '.h5')

