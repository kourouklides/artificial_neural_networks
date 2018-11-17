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

# %%


def none_or_int(input_arg):
    """
    Utility to return None or int value
    """
    if input_arg == 'None':
        value = None
    else:
        value = int(input_arg)

    return value


# %%


def none_or_float(input_arg):
    """
    Utility to return None or float value
    """
    if input_arg == 'None':
        value = None
    else:
        value = float(input_arg)

    return value
