"""

Utility to download the CIFAR-10 dataset

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


def download_cifar_10(new_dir=os.getcwd()):
    """

    Main function

    """
    # %%
    # IMPORTS

    os.chdir(new_dir)

    # code repository sub-package imports
    from artificial_neural_networks.utils.data_utils import download_dataset

    # %%

    file_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    file_name = 'cifar-10-python.tar.gz'

    file_path = download_dataset(file_url, file_name)

    # %%

    return file_path


# %%

if __name__ == '__main__':
    dataset_path = download_cifar_10('../../')
