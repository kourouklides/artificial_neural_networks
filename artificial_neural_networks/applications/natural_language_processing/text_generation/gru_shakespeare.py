"""

Model: GRU
Mehtod: Truncated Backpropagation Through Time (TBPTT)
Architecture: Recurrent Neural Network

Dataset: Shakespeare
Task: Text generation

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
import os
import time


# third-party imports
import numpy as np
import tensorflow as tf


# Enable eager execution in TensorFlow
tf.compat.v1.enable_eager_execution()


# %%
# SETTINGS

parser = argparse.ArgumentParser()

# General settings
parser.add_argument('--verbose', type=int, default=1)

args = parser.parse_args()


seq_length = 100
BATCH_SIZE = 256
embedding_dim = 256
rnn_units = 256
EPOCHS = 10


# %%
# Utility functions

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]

    return input_text, target_text


# %%
# Model

def build_model(vocab_size, embedding_dim, rnn_units, batch_size, rnn):
    model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                      batch_input_shape=[batch_size, None]),
            rnn(rnn_units,
                return_sequences=True,
                recurrent_initializer='glorot_uniform',
                stateful=True),
            tf.keras.layers.Dense(vocab_size)])

    return model


# %%
# Load the Shakespeare dataset

path_to_file = tf.keras.utils.get_file(
        'shakespeare.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

if args.verbose > 0:
    # length of text is the number of characters in it
    print('Length of text: {} characters'.format(len(text)))

    # The unique characters in the file
    vocab = sorted(set(text))
    print('{} unique characters'.format(len(vocab)))


# %%
# Training set

BUFFER_SIZE = 10000
examples_per_epoch = len(text)//seq_length
steps_per_epoch = examples_per_epoch // BATCH_SIZE
vocab_size = len(vocab)

# Creating a mapping from unique characters to indices
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

# Create training examples and targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

dataset = sequences.map(split_input_target)

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


# %%
# Model hyperparameters and ANN Architecture

if tf.test.is_gpu_available():
    RNN = tf.keras.layers.CuDNNGRU
else:
    import functools
    RNN = functools.partial(tf.keras.layers.GRU, recurrent_activation='sigmoid')

model = build_model(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=BATCH_SIZE,
        rnn=RNN)

model.summary()


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(), loss=loss)


# %%
# Save trained models for every epoch

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)


# %%
# TRAINING PHASE

history = model.fit(dataset.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
                    callbacks=[checkpoint_callback])


# %%

tf.train.latest_checkpoint(checkpoint_dir)

model_2 = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model_2.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model_2.build(tf.TensorShape([1, None]))

model_2.summary()


# %%
# TESTING PHASE

def generate_text(model, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 1000

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a multinomial distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


print(generate_text(model_2, start_string=u"ROMEO: "))

# %%
# TRAINING PHASE

model_3 = build_model(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=BATCH_SIZE,
        rnn=RNN)

optimizer = tf.train.AdamOptimizer()

# Training step
EPOCHS = 1

for epoch in range(EPOCHS):
    start = time.time()

    # initializing the hidden state at the start of every epoch
    # initially hidden is None
    hidden = model_3.reset_states()

    for (batch_n, (inp, target)) in enumerate(dataset):
        with tf.GradientTape() as tape:
            # feeding the hidden state back into the model
            # This is the interesting step
            predictions = model_3(inp)
            loss = tf.losses.sparse_softmax_cross_entropy(target, predictions)

        grads = tape.gradient(loss, model_3.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_3.trainable_variables))

        if batch_n % 5 == 0:
            template = 'Epoch {} Batch {} Loss {:.4f}'
            print(template.format(epoch+1, batch_n, loss))

    # saving (checkpoint) the model every 1 epoch
    if (epoch + 1) % 1 == 0:
        model_3.save_weights(checkpoint_prefix.format(epoch=epoch))

    print('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

model_3.save_weights(checkpoint_prefix.format(epoch=epoch))


# %%
# Examples & Debugging

for i in char_dataset.take(5):
    print(idx2char[i.numpy()])

for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))

for input_example, target_example in dataset.take(1):
    print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    print('Target data:', repr(''.join(idx2char[target_example.numpy()])))


for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

example_batch_loss = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape,
      " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())
