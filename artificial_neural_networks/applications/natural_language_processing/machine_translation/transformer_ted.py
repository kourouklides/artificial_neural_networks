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

MAX_LENGTH = 40
BATCH_SIZE = 256



# %%
# Utility functions

def encode(lang1, lang2):
    """
    Add a start and end token to the input and target
    """
    lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
        lang1.numpy()) + [tokenizer_pt.vocab_size+1]

    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
        lang2.numpy()) + [tokenizer_en.vocab_size+1]

    return lang1, lang2


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)


def tf_encode(lang1, lang2):
    """
    Receive an eager tensor having a numpy attribute that contains the string value
    """
    return tf.py_function(encode, [lang1, lang2], [tf.int64, tf.int64])


# %%
# Seq2Seq model


# %%
# Load the TED Talks dataset

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)


# %%
# Training and Validation sets

train_examples, val_examples = examples['train'], examples['validation']

tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for pt, en in train_examples), target_vocab_size=2**13)

tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)

BUFFER_SIZE = 20000

train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)

# cache the dataset to memory to get a speedup while reading from it.
train_dataset = train_dataset.cache()

train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(
        BATCH_SIZE, padded_shapes=([-1], [-1]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(
        BATCH_SIZE, padded_shapes=([-1], [-1]))





# %%
# Examples & Debugging

sample_string = 'Transformer is awesome.'

tokenized_string = tokenizer_en.encode(sample_string)
print('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer_en.decode(tokenized_string)
print('The original string: {}'.format(original_string))

assert original_string == sample_string


for ts in tokenized_string:
    print('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))


pt_batch, en_batch = next(iter(val_dataset))
pt_batch, en_batch

















