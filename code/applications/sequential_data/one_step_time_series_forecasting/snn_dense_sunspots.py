"""

Model: Standard Neural Network (SNN) with dense (i.e. fully connected) layers
Method: Backpropagation
Architecture: Feedforward Neural Network

Dataset: Monthly sunspots
Task: One-step Ahead Forecasting of Univariate Time Series (Univariate Regression)

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
import os

# %%


if __name__ == '__main__':
    os.chdir('../../../../../')

    # code repository sub-package imports
    from artificial_neural_networks.code.architectures.recurrent_neural_networks.LSTM. \
        snn_dense_sunspots import snn_dense_sunspots

    model_snn_dense = snn_dense_sunspots()
