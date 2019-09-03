#!/usr/bin/env python
"""
Bayesian neural network using mean-field variational inference
(see, e.g., Blundell et al. (2015); Kucukelbir et al. (2016)).

    Author: Ioannis Kourouklides

"""
#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal, Categorical

import argparse

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_iter', type=int, default=1000)
parser.add_argument('--n_samples', type=int, default=1)
parser.add_argument('--n_posterior_samples', type=int, default=10)
args = parser.parse_args()
print(args)

# Load MNIST data
mnist = np.load('mnist.npz')
trainx = (256/126)*mnist['x_train'].astype(np.float32)
trainy = mnist['y_train'].astype(np.int32)
testx = (256/126)*mnist['x_test'].astype(np.float32)
testy = mnist['y_test'].astype(np.int32)
validx = (256/126)*mnist['x_valid'].astype(np.float32)
validy = mnist['y_valid'].astype(np.int32)

# For reproducibility
#np.random.seed(args.seed)
#tf.set_random_seed(args.seed))
#ed.set_seed(args.seed))

train_N = trainx.shape[0]

# Network Parameters
n_D = trainx.shape[1] # number of features / dimensions
n_out = np.unique(trainy).shape[0] # number of classes
#N = [n_D, 400, 400, n_out] # layer sizes
N = [n_D, 11, 11, n_out] # layer sizes

# Define network (with deterministic activation/transfer function)
def neural_network(x,w,b):
    L = len(N)-1
    h = x
    for i in range(1,L):
        h = tf.nn.relu(tf.add(tf.matmul(h, w[i-1]), b[i-1])) # hidden layer i
    out = tf.nn.softmax(tf.add(tf.matmul(h, w[L-1]), b[L-1])) # output layer
    
    return out

# MODEL
"""
Bayesian neural network for classification.

Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. "Weight 
Uncertainty in Neural Networks." ICML 2015.
arXiv:1505.05424
"""
w = []
b = []
qw = []
qb = []
latent_vars = {}
for i in range(1,len(N)):
    m, n = N[i-1], N[i]
    
    # Probability model (uncertainty in weights)
    w_mu = tf.zeros([m, n])
    w_sigma = tf.ones([m, n])
    b_mu = tf.zeros(n)
    b_sigma = tf.ones(n)
    
    # Variational model
    qw_mu = tf.Variable(tf.random_normal([m, n]))
    qw_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([m, n])))
    qb_mu = tf.Variable(tf.random_normal([n]))
    qb_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([n])))
    
    w += [Normal(mu=w_mu, sigma=w_sigma)]
    b += [Normal(mu=b_mu, sigma=b_sigma)]
    qw += [Normal(mu=qw_mu, sigma=qw_sigma)]
    qb += [Normal(mu=qb_mu, sigma=qb_sigma)]
    
    latent_vars[w[i-1]] = qw[i-1]
    latent_vars[b[i-1]] = qb[i-1]

x = tf.convert_to_tensor(trainx, dtype=tf.float32)
y = Categorical(neural_network(x,w,b))

# INFERENCE
data = {y: trainy}
inference = ed.MFVI(latent_vars, data)

#%%
# Sample functions from variational model to then evaluate predictions
x_ = tf.placeholder(tf.float32, shape=[None, n_D])
y_ = tf.placeholder(tf.float32, shape=[None, n_out])

n_posterior_samples = args.n_posterior_samples

# Monte Carlo estimate of the mean of the posterior predictive
mus = []
for _ in range(n_posterior_samples):
    qw_sample = []
    qb_sample = []
    for i in range(1,len(N)):
        qw_sample += [qw[i-1].sample()]
        qb_sample += [qb[i-1].sample()]
   
    mus += [neural_network(x_, qw_sample, qb_sample)]

draws = tf.pack(mus)
draws_avg = tf.reduce_mean(mus,0)

# Evaluation
y_true = tf.argmax(y_,1)
y_pred = tf.argmax(draws_avg,1)
correct_prediction = tf.equal(y_true, y_pred)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#%%
# Initialization
#optimizer = tf.train.RMSPropOptimizer(1e-2, epsilon=1.0)
optimizer = tf.train.RMSPropOptimizer(1e-1, epsilon=1e-4)

inference.initialize(optimizer=optimizer, n_samples=args.n_samples, n_print=10,
                     n_iter=args.n_iter, scale={x: float(train_N) / 100})

sess = ed.get_session()
init = tf.initialize_all_variables()
init.run()

#%%
# Training iterations
for t in range(inference.n_iter):
    print('Iteration %d' % t)

    info_dict = inference.update()
    print('Train loss = %f' % info_dict['loss'])

    # Perform evaluation every 10 epochs
    if t % inference.n_print == 0:
        test_acc = accuracy.eval(feed_dict={x_: testx, 
                                             y_: tf.one_hot(testy,n_out).eval()})
        test_err = 1 - test_acc
        print('Test error is %f' % test_err)
        
        train_acc = accuracy.eval(feed_dict={x_: trainx,
                                             y_: tf.one_hot(trainy,n_out).eval()})
        train_err = 1 - train_acc
        print('Training error is %f' % train_err)
        
        valid_acc = accuracy.eval(feed_dict={x_: np.float32(validx),
                                             y_: tf.one_hot(validy,n_out).eval()})
        valid_err = 1 - valid_acc
        print('Validation error is %f' % valid_err)

