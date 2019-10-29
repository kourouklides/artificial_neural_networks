"""

Model: Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors (SARIMAX)
Method: Maximum Likelihood Estimation (MLE) via Kalman filter

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
from math import sqrt
import os
import random as rn
from timeit import default_timer as timer

# third-party imports
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

# %%


def sarimax_sunspots(args):
    """
    Main function
    """
    # %%
    # IMPORTS

    # code repository sub-package imports
    from artificial_neural_networks.utils.download_monthly_sunspots import \
        download_monthly_sunspots
    from artificial_neural_networks.utils.generic_utils import affine_transformation
    from artificial_neural_networks.utils.vis_utils import regression_figs

    # %%

    if args.verbose > 0:
        print(args)

    # For reproducibility
    if args.reproducible:
        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(args.seed)
        rn.seed(args.seed)

    # %%
    # Load the Monthly sunspots dataset

    sunspots_path = download_monthly_sunspots()
    sunspots = np.genfromtxt(
        fname=sunspots_path, dtype=np.float32, delimiter=",", skip_header=1, usecols=1)

    # %%
    # Train-Test split

    L_series = len(sunspots)

    split_ratio = 2 / 3  # between zero and one
    n_split = int(L_series * split_ratio)

    train_y = sunspots[:n_split]
    test_y = sunspots[n_split:]

    # %%
    # PREPROCESSING STEP

    scaling_factor = args.scaling_factor
    translation = args.translation

    n_train = train_y.shape[0]  # number of training examples/samples

    # Apply preprocessing
    train_y_ = affine_transformation(train_y, scaling_factor, translation)
    test_y_ = affine_transformation(test_y, scaling_factor, translation)

    # %%
    # Model hyperparameters

    optimizer = 'lbfgs'
    # optimizer = 'powell'

    maxiter = 15
    # maxiter = 50

    order = (args.autoregressive, args.integrated, args.moving_average)
    seasonal_order = (0, 0, 0, 0)
    trend = 'ct'

    # %%
    # TRAINING PHASE

    train_outliers = np.zeros(n_train)

    train_model = SARIMAX(train_y_, order=order, seasonal_order=seasonal_order,
                          exog=train_outliers, trend=trend)

    if args.time_training:
        start = timer()

    fitted_params = None

    for i in range(1):
        model_fit = train_model.fit(start_params=fitted_params, method=optimizer, maxiter=maxiter)
        fitted_params = model_fit.params

        if args.verbose > 0:
            print('All parameters:')
            print(fitted_params)

    if args.time_training:
        end = timer()
        duration = end - start
        print('Total time for training (in seconds):')
        print(duration)

    if args.verbose > 0:
        print(model_fit.summary())

    def model_predict(y):
        """
        Predict using the SARIMAX Model (One-step ahead Forecasting)
        """
        n_y = y.shape[0]

        pred_outliers = np.zeros(n_y)
        pred_model = SARIMAX(y, order=order, seasonal_order=seasonal_order,
                             exog=pred_outliers, trend=trend)
        y_pred = pred_model.filter(fitted_params).get_prediction().predicted_mean

        return y_pred

    # %%
    # TESTING PHASE

    # Predict preprocessed values
    train_y_pred_ = model_predict(train_y_)
    test_y_pred_ = model_predict(test_y_)

    # Remove preprocessing
    train_y_pred = affine_transformation(train_y_pred_, scaling_factor, translation, inverse=True)
    test_y_pred = affine_transformation(test_y_pred_, scaling_factor, translation, inverse=True)

    train_rmse = sqrt(mean_squared_error(train_y, train_y_pred))
    train_mae = mean_absolute_error(train_y, train_y_pred)
    train_r2 = r2_score(train_y, train_y_pred)

    test_rmse = sqrt(mean_squared_error(test_y, test_y_pred))
    test_mae = mean_absolute_error(test_y, test_y_pred)
    test_r2 = r2_score(test_y, test_y_pred)

    if args.verbose > 0:
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

    if args.plot:
        regression_figs(train_y=train_y, train_y_pred=train_y_pred,
                        test_y=test_y, test_y_pred=test_y_pred)

    # %%

    model = {}
    model['params'] = fitted_params
    model['hyperparams'] = {}
    model['hyperparams']['order'] = order
    model['hyperparams']['seasonal_order'] = seasonal_order
    model['hyperparams']['trend'] = trend

    return model


# %%


if __name__ == '__main__':

    # %%
    # IMPORTS

    os.chdir('../../../../../')

    # %%
    # SETTINGS

    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--reproducible', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--time_training', type=bool, default=True)
    parser.add_argument('--plot', type=bool, default=False)

    # Settings for preprocessing and hyperparameters
    parser.add_argument('--scaling_factor', type=float, default=1)
    parser.add_argument('--translation', type=float, default=0)
    parser.add_argument('--autoregressive', type=float, default=20)
    parser.add_argument('--integrated', type=float, default=0)
    parser.add_argument('--moving_average', type=float, default=2)

    args = parser.parse_args()

    # %%
    # MODEL

    model_sarimax_sunspots = sarimax_sunspots(args)
