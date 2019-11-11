"""

Model: Bidirectional Long short-term memory (LSTM) with dropout layers
Mehtod: Truncated Backpropagation Through Time (TBPTT)
Architecture: Recurrent Neural Network

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

    os.chdir('../../../../')

    # code repository sub-package imports
    from artificial_neural_networks.architectures.recurrent_neural_networks.LSTM. \
        bi_lstm_dropout_sunspots import bi_lstm_dropout_sunspots
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
    parser.add_argument('--look_back', type=int, default=3)
    parser.add_argument('--scaling_factor', type=float, default=(1 / 780))
    parser.add_argument('--translation', type=float, default=0)
    parser.add_argument('--layer_size', type=int, default=4)
    parser.add_argument('--n_epochs', type=int, default=13)
    parser.add_argument('--batch_size', type=none_or_int, default=1)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lrearning_rate', type=float, default=1e-3)
    parser.add_argument('--epsilon', type=none_or_float, default=None)
    parser.add_argument('--dropout_rate_input', type=float, default=0.01)
    parser.add_argument('--dropout_rate_hidden', type=float, default=0.01)

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

    model_bi_lstm_dense = bi_lstm_dropout_sunspots(args)
