"""

Model: Long short-term memory (LSTM) with dense (i.e. fully connected) layers
Mehtod: Truncated Backpropagation Through Time (TBPTT)
Architecture: Recurrent Neural Network

Dataset: Monthly sunspots
Task: Time Series Forecasting (Univariate Regression)

    Author: Ioannis Kourouklides, www.kourouklides.com
    License: https://github.com/kourouklides/artificial_neural_networks/blob/master/LICENSE

"""
# %%
# Python configurations

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# %%


if __name__ == '__main__':
    os.chdir('../../../../../')

    from artificial_neural_networks.code.architectures.recurrent_neural_networks.LSTM. \
        lstm_dense_sunsposts import lstm_dense_sunsposts

    model = lstm_dense_sunsposts()
