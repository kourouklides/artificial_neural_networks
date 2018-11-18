"""

Model: Bidirectional Long short-term memory (LSTM) with dropout layers
Mehtod: Truncated Backpropagation Through Time (TBPTT)
Architecture: Recurrent Neural Network

Dataset: Monthly sunspots
Task: Time Series Forecasting (Univariate Regression)

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
        bi_lstm_dropout_sunspots import bi_lstm_dropout_sunspots

    model_bi_lstm_dense = bi_lstm_dropout_sunspots()
