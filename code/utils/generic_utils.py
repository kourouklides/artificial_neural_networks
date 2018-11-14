"""

Utility to download a dataset

    Author: Ioannis Kourouklides, www.kourouklides.com
    License: https://github.com/kourouklides/artificial_neural_networks/blob/master/LICENSE


"""
# %%

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def none_or_int(value):
    if value == 'None':
        return None
    else:
        return int(value)


def none_or_float(value):
    if value == 'None':
        return None
    else:
        return float(value)
