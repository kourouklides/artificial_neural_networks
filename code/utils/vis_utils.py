"""

Utility to download a dataset

    Author: Ioannis Kourouklides, www.kourouklides.com
    License: https://github.com/kourouklides/artificial_neural_networks/blob/master/LICENSE


"""
# %%
# IMPORTS

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# standard library imports
import itertools

# third-party imports
import matplotlib.pyplot as plt
import numpy as np

# %%


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Utility to plot the Confusion matrix
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


# %%


def epoch_plot(x_values, y_train, y_test, y_axis):
    """
    Utility to plot Y vs epoch, where Y should be provided
    """
    plt.figure()
    plt.plot(x_values, y_train, 'r--')
    plt.plot(x_values, y_test, 'b-')
    plt.legend(['Training ' + y_axis, 'Test ' + y_axis])
    plt.xlabel('Epoch')
    plt.ylabel(y_axis)
    plt.show()


# %%


def regression_figs(train_y, train_y_pred, test_y, test_y_pred):
    """
    Utility to plot regression figures
    """
    plt.figure()
    plt.plot(train_y)
    plt.plot(train_y_pred)
    plt.title('Time Series of the training set')
    plt.show()

    plt.figure()
    plt.plot(test_y)
    plt.plot(test_y_pred)
    plt.title('Time Series of the test set')
    plt.show()

    train_errors = train_y - train_y_pred
    plt.figure()
    plt.hist(train_errors, bins='auto')
    plt.title('Histogram of training errors')
    plt.show()

    test_errors = test_y - test_y_pred
    plt.figure()
    plt.hist(test_errors, bins='auto')
    plt.title('Histogram of test errors')
    plt.show()

    plt.figure()
    plt.scatter(x=train_y, y=train_y_pred, edgecolors=(0, 0, 0))
    plt.plot([train_y.min(), train_y.max()], [train_y.min(), train_y.max()], 'k--', lw=4)
    plt.title('Predicted vs Actual for training set')
    plt.show()

    plt.figure()
    plt.scatter(x=test_y, y=test_y_pred, edgecolors=(0, 0, 0))
    plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=4)
    plt.title('Predicted vs Actual for test set')
    plt.show()

    plt.figure()
    plt.scatter(x=train_y_pred, y=train_errors, edgecolors=(0, 0, 0))
    plt.plot([train_y.min(), train_y.max()], [0, 0], 'k--', lw=4)
    plt.title('Residuals vs Predicted for training set')
    plt.show()

    plt.figure()
    plt.scatter(x=test_y_pred, y=test_errors, edgecolors=(0, 0, 0))
    plt.plot([test_y.min(), test_y.max()], [0, 0], 'k--', lw=4)
    plt.title('Residuals vs Predicted for test set')
    plt.show()
