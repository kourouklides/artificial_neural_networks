"""

Model: Standard Neural Network (SNN) with dense (i.e. fully connected) layers
Method: Backpropagation
Architecture: Feedforward Neural Network

Dataset: Monthly sunspots
Task: One-step ahead Forecasting of Univariate Time Series (Univariate Regression)

    Author: Ioannis Kourouklides, www.kourouklides.com
    License:
        https://github.com/kourouklides/artificial_neural_networks/blob/master/LICENSE

"""
# %%
# IMPORTS

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# standard library imports
import argparse
import os

# %%


if __name__ == '__main__':

    # %%
    # IMPORTS

    os.chdir('../../../../../')

    # code repository sub-package imports
    from artificial_neural_networks.architectures.feedforward_neural_networks.\
        standard_neural_networks.snn_dense_sunspots import snn_dense_sunspots
    from artificial_neural_networks.utils.generic_utils import none_or_int, none_or_float

    # %%
    # SETTINGS

    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--reproducible', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--time_training', type=bool, default=True)
    parser.add_argument('--plot', type=bool, default=False)

    # Settings for preprocessing and hyperparameters
    parser.add_argument('--look_back', type=int, default=10)
    parser.add_argument('--scaling_factor', type=float, default=(1 / 780))
    parser.add_argument('--translation', type=float, default=0)
    parser.add_argument('--same_size', type=bool, default=True)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--layer_size', type=int, default=128)
    parser.add_argument('--explicit_layer_sizes', nargs='*', type=int, default=[128, 128])
    parser.add_argument('--n_epochs', type=int, default=7)
    parser.add_argument('--batch_size', type=none_or_int, default=1)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lrearning_rate', type=float, default=1e-3)
    parser.add_argument('--epsilon', type=none_or_float, default=None)

    # Settings for saving the model
    parser.add_argument('--save_architecture', type=bool, default=True)
    parser.add_argument('--save_last_weights', type=bool, default=True)
    parser.add_argument('--save_last_model', type=bool, default=True)
    parser.add_argument('--save_models', type=bool, default=False)
    parser.add_argument('--save_weights_only', type=bool, default=False)
    parser.add_argument('--save_best', type=bool, default=True)

    args = parser.parse_args()

    # %%
    # MODEL

    model_snn_dense = snn_dense_sunspots(args)
