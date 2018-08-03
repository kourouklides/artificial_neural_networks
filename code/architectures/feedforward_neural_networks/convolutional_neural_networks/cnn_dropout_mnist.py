"""

Model: Convolutional Neural Network (SNN) with dense (i.e. fully connected) layers
Mehtod: Backpropagation

ANN Architecture: Feedforward Neural Network
Dataset: MNIST

    Author: Ioannis Kourouklides, www.kourouklides.com

"""
#%% 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from keras import optimizers
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.utils import to_categorical

import argparse

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--verbose', type = int, default = 1)
parser.add_argument('--reproducible', type = bool, default = True)
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--n_layers', type = int, default = 2)
parser.add_argument('--layer_size', type = int, default = 512)
parser.add_argument('--n_epochs', type = int, default = 20)
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--dropout_rate', type = int, default = 0.2)
args = parser.parse_args()

if (args.verbose > 0):
    print(args)

# For reproducibility
if (args.reproducible):
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

#%% 
# PREPROCESSING STEP
scaling_factor = (1/255) 

# Load MNIST data
mnist_path = '../../../data/mnist.npz'
mnist = np.load(mnist_path)
train_x = scaling_factor * mnist['x_train'].astype(np.float32)
train_y = mnist['y_train'].astype(np.int32)
test_x = scaling_factor * mnist['x_test'].astype(np.float32)
test_y = mnist['y_test'].astype(np.int32)
mnist.close()

img_width = train_x.shape[1]
img_length = train_x.shape[2]

n_train = train_x.shape[0] # number of training examples/samples
n_test = test_x.shape[0] # number of test examples/samples
n_in = img_width * img_length # number of features / dimensions
n_out = np.unique(train_y).shape[0] # number of classes/labels

# Reshape training and test sets
train_x = train_x.reshape(n_train, n_in)
test_x = test_x.reshape(n_test, n_in)

# Convert class vectors to binary class matrices (i.e. One hot encoding)
one_hot = False
if (one_hot):
    train_y = to_categorical(train_y, n_out)
    test_y = to_categorical(test_y, n_out)

# Model hyperparameters
N = []
N.append(n_in) #input layer
for i in range(args.n_layers):
    N.append(args.layer_size) # hidden layer i
N.append(n_out) # output layer
lrearning_rate = 1e-3
epsilon = None
optimizer = optimizers.RMSprop(lr=lrearning_rate,epsilon=epsilon)
# optimizer = optimizers.Adam(lr=lrearning_rate,epsilon=epsilon)

# ANN Architecture
x = Input(shape=(n_in,)) #input layer
h = x
for i in range(args.n_layers):
    h = Dense(args.layer_size, activation = 'relu')(h) # hidden layer i
    h = Dropout(args.dropout_rate)(h)
out = Dense(n_out, activation = 'softmax')(h) # output layer

model = Model(inputs = x, outputs = out)

if (args.verbose > 0):
    model.summary()

# loss_function = 'categorical_crossentropy'
if (one_hot):
    loss_function = 'categorical_crossentropy'
else:
    loss_function = 'sparse_categorical_crossentropy'

metrics=['accuracy']

model.compile(optimizer = optimizer, \
              loss = loss_function, \
              metrics = metrics)

#%% 
# TRAINING PHASE
model_history = model.fit(x = train_x, y = train_y,
                          batch_size = args.batch_size, \
                          epochs = args.n_epochs, \
                          verbose = args.verbose, \
                          validation_data = (test_x, test_y))

#%% 
# VALIDATION PHASE
score = model.evaluate(x = test_x, y = test_y, verbose = args.verbose)

if (args.verbose > 0):
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

