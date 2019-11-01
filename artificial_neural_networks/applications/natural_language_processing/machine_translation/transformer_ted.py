"""

Model: Transformer (i.e. FNN with self-attention), Seq2Seq (i.e. Encoder-Decoder)
Mehtod: Backpropagation
Architecture: Recurrent Neural Network

Dataset: TED Talks
Task: Machine Translation

    Author: Ioannis Kourouklides, www.kourouklides.com
    License:
        https://github.com/kourouklides/artificial_neural_networks/blob/master/LICENSE

"""
# %%
# IMPORTS

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


# standard library imports
import argparse
import time

# third-party imports
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


# Enable eager execution in TensorFlow
tf.compat.v1.enable_eager_execution()


# %%
# SETTINGS

parser = argparse.ArgumentParser()

# General settings
parser.add_argument('--verbose', type=int, default=1)

args = parser.parse_args()

# %%
# Utility functions


# %%
# Seq2Seq model


# %%
# Load the TED Talks dataset

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)

# %%
# Training and Validation sets

train_examples, val_examples = examples['train'], examples['validation']


















