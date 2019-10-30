"""

Model: Long short-term memory (LSTM) with dense (i.e. fully connected) layers
Mehtod: Truncated Backpropagation Through Time (TBPTT)
Architecture: Recurrent Neural Network

Dataset: Monthly sunspots
Task: One-step Ahead Forecasting of Univariate Time Series (Univariate Regression) with Deseasoning

    Author: Ioannis Kourouklides, www.kourouklides.com
    License:
        https://github.com/kourouklides/artificial_neural_networks/blob/master/LICENSE

"""
# %%
# Python configurations

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# standard library imports
import argparse
from math import sqrt
import os
import random as rn
from timeit import default_timer as timer

# third-party imports
from keras import backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, LSTM
from keras.models import Model
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

# %%


def lstm_dense_sunspots(args):
    """
    Main function
    """
    # %%
    # IMPORTS

    # code repository sub-package imports
    from artificial_neural_networks.utils.download_monthly_sunspots import \
        download_monthly_sunspots
    from artificial_neural_networks.utils.generic_utils import save_regress_model, \
        series_to_supervised, affine_transformation
    from artificial_neural_networks.utils.vis_utils import regression_figs

    # %%

    if args.verbose > 0:
        print(args)

    # For reproducibility
    if (args.reproducible):
        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(args.seed)
        rn.seed(args.seed)
        tf.set_random_seed(args.seed)
        sess = tf.Session(graph=tf.get_default_graph())
        K.set_session(sess)
        # print(hash("keras"))

    # %%
    # Load the Monthly sunspots dataset

    sunspots_path = download_monthly_sunspots()
    sunspots = np.genfromtxt(
        fname=sunspots_path, dtype=np.float32, delimiter=",", skip_header=1, usecols=1)

    # %%
    # Train-Test split

    L_series = len(sunspots)

    split_ratio = 2/3  # between zero and one
    n_split = int(L_series * split_ratio)

    look_back = args.look_back

    diff = args.diff

    train = sunspots[:n_split]
    test = sunspots[n_split - diff - look_back:]

    train_y = train
    test_y = test[diff + look_back:]

    # Apply diferencing for Seasonality Adjustment to make the it stationary
    d_train = train[diff:] - train[:-diff]
    d_test = test[diff:] - test[:-diff]

    train_first = train[look_back:look_back + diff]
    test_first = test[look_back:look_back + diff]

    d_train_x, d_train_y = series_to_supervised(d_train, look_back)
    d_test_x, d_test_y = series_to_supervised(d_test, look_back)

    # %%
    # PREPROCESSING STEP

    scaling_factor = args.scaling_factor
    translation = args.translation

    n_train = d_train_x.shape[0]  # number of training examples/samples
    n_test = d_test_x.shape[0]  # number of test examples/samples

    n_in = d_train_x.shape[1]  # number of features / dimensions
    n_out = 1  # number of classes/labels

    # Reshape training and test sets
    d_train_x = d_train_x.reshape(n_train, n_in, 1)
    d_test_x = d_test_x.reshape(n_test, n_in, 1)

    # Apply preprocessing
    d_train_x_ = affine_transformation(d_train_x, scaling_factor, translation)
    d_train_y_ = affine_transformation(d_train_y, scaling_factor, translation)
    d_test_x_ = affine_transformation(d_test_x, scaling_factor, translation)
    d_test_y_ = affine_transformation(d_test_y, scaling_factor, translation)

    # %%
    # Model hyperparameters and ANN Architecture

    x = Input(shape=(n_in, 1))  # input layer
    h = x

    h = LSTM(units=args.layer_size)(h)  # hidden layer

    out = Dense(units=n_out, activation=None)(h)  # output layer

    model = Model(inputs=x, outputs=out)

    if (args.verbose > 0):
        model.summary()

    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

    loss_function = root_mean_squared_error

    metrics = ['mean_absolute_error', 'mean_absolute_percentage_error']

    lr = args.lrearning_rate
    epsilon = args.epsilon
    optimizer_selection = {
        'Adadelta':
        optimizers.Adadelta(lr=lr, rho=0.95, epsilon=epsilon, decay=0.0),
        'Adagrad':
        optimizers.Adagrad(lr=lr, epsilon=epsilon, decay=0.0),
        'Adam':
        optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=epsilon, decay=0.0, amsgrad=False),
        'Adamax':
        optimizers.Adamax(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=epsilon, decay=0.0),
        'Nadam':
        optimizers.Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=epsilon, schedule_decay=0.004),
        'RMSprop':
        optimizers.RMSprop(lr=lr, rho=0.9, epsilon=epsilon, decay=0.0),
        'SGD':
        optimizers.SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)
    }

    optimizer = optimizer_selection[args.optimizer]

    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

    # %%
    # Save trained models for every epoch

    models_path = r'artificial_neural_networks/trained_models/'
    model_name = 'sunspots_lstm_dense_des'
    weights_path = models_path + model_name + '_weights'
    model_path = models_path + model_name + '_model'
    file_suffix = '_{epoch:04d}_{val_loss:.4f}_{val_mean_absolute_error:.4f}'

    if (args.save_weights_only):
        file_path = weights_path
    else:
        file_path = model_path

    file_path += file_suffix

    monitor = 'val_loss'

    if args.save_models:
        checkpoint = ModelCheckpoint(
            file_path + '.h5',
            monitor=monitor,
            verbose=args.verbose,
            save_best_only=args.save_best,
            mode='auto',
            save_weights_only=args.save_weights_only)
        callbacks = [checkpoint]
    else:
        callbacks = []

    # %%
    # TRAINING PHASE

    if args.time_training:
        start = timer()

    model.fit(
        x=d_train_x_,
        y=d_train_y_,
        validation_data=(d_test_x_, d_test_y_),
        batch_size=args.batch_size,
        epochs=args.n_epochs,
        verbose=args.verbose,
        callbacks=callbacks)

    if args.time_training:
        end = timer()
        duration = end - start
        print('Total time for training (in seconds):')
        print(duration)

    # %%
    # TESTING PHASE

    # Predict values (with preprocessing and differencing)
    d_train_y_pred_ = model.predict(d_train_x_)[:, 0]
    d_test_y_pred_ = model.predict(d_test_x_)[:, 0]

    # Remove preprocessing
    d_train_y_pred = affine_transformation(d_train_y_pred_, scaling_factor, translation,
                                           inverse=True)
    d_test_y_pred = affine_transformation(d_test_y_pred_, scaling_factor, translation,
                                          inverse=True)

    def remove_diff(d_y, y_first):
        """
        Remove differencing
        """
        y = np.insert(d_y, 0, y_first)
        d = y_first.shape[0]
        n_y = y.shape[0]
        n_zeros = d - (n_y % d)
        d_y_pad = np.insert(np.zeros(n_zeros), 0, y)
        n_rows = len(d_y_pad)//d
        n_cols = d
        d_y_reshaped = d_y_pad.reshape(n_rows, n_cols)
        y_pad = d_y_reshaped.cumsum(axis=0).reshape(n_rows*n_cols)
        y = y_pad[:-n_zeros]
        y[:d] = np.zeros(d)

        return y

    # Remove differencing
    train_y_pred = np.concatenate((np.zeros(look_back), remove_diff(d_train_y_pred, train_first)))
    test_y_pred = remove_diff(d_test_y_pred, test_first)[diff:]

    train_rmse = sqrt(mean_squared_error(train_y, train_y_pred))
    train_mae = mean_absolute_error(train_y, train_y_pred)
    train_r2 = r2_score(train_y, train_y_pred)

    test_rmse = sqrt(mean_squared_error(test_y, test_y_pred))
    test_mae = mean_absolute_error(test_y, test_y_pred)
    test_r2 = r2_score(test_y, test_y_pred)

    if (args.verbose > 0):
        print('Train RMSE: %.4f ' % (train_rmse))
        print('Train MAE: %.4f ' % (train_mae))
        print('Train (1 - R_squared): %.4f ' % (1.0 - train_r2))
        print('Train R_squared: %.4f ' % (train_r2))
        print('')
        print('Test RMSE: %.4f ' % (test_rmse))
        print('Test MAE: %.4f ' % (test_mae))
        print('Test (1 - R_squared): %.4f ' % (1.0 - test_r2))
        print('Test R_squared: %.4f ' % (test_r2))

    # %%
    # Data Visualization

    if (args.plot):
        regression_figs(train_y=train_y, train_y_pred=train_y_pred,
                        test_y=test_y, test_y_pred=test_y_pred)

    # %%
    # Save the architecture and the lastly trained model

    save_regress_model(model, models_path, model_name, weights_path, model_path, file_suffix,
                       test_rmse, test_mae, args)

    # %%

    return model


# %%


if __name__ == '__main__':

    # %%
    # IMPORTS

    os.chdir('../../../../')

    # code repository sub-package imports
    from artificial_neural_networks.utils.generic_utils import none_or_int, none_or_float

    # %%
    # SETTINGS
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--reproducible', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--time_training', type=bool, default=True)
    parser.add_argument('--plot', type=bool, default=True)

    # Settings for preprocessing and hyperparameters
    parser.add_argument('--look_back', type=int, default=10)
    parser.add_argument('--scaling_factor', type=float, default=(1 / 780))
    parser.add_argument('--translation', type=float, default=0)
    parser.add_argument('--layer_size', type=int, default=4)
    parser.add_argument('--n_epochs', type=int, default=13)
    parser.add_argument('--batch_size', type=none_or_int, default=1)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lrearning_rate', type=float, default=1e-3)
    parser.add_argument('--epsilon', type=none_or_float, default=None)
    parser.add_argument('--diff', type=int, default=126)

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

    model_lstm_dense = lstm_dense_sunspots(args)
