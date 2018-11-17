"""

Utilities related to datasets

    Author: Ioannis Kourouklides, www.kourouklides.com
    License: https://github.com/kourouklides/artificial_neural_networks/blob/master/LICENSE


"""
# %%

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves.urllib.request import urlretrieve

import os
from os.path import dirname as up

# %%


def download_dataset(file_url, file_name):
    """
    Utility to download a dataset
    """

    new_dir = up(up(up(up(os.path.abspath(__file__)))))
    os.chdir(new_dir)

    file_path = r'artificial_neural_networks/datasets/' + file_name
    exists = os.path.isfile(file_path)
    if exists:
        print(file_name + ' already exists.')
        print('You have to delete it first, if you want to re-download it.')
    else:
        urlretrieve(file_url, file_path)
        print(file_name + ' was downloaded succesfully.')

    return file_path
