"""

Utility to download the IMDB dataset

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


def download_imdb(new_dir=os.getcwd()):
    """

    Main function

    """
    # %%
    # IMPORTS

    os.chdir(new_dir)

    # code repository sub-package imports
    from artificial_neural_networks.utils.data_utils import download_dataset

    # %%

    file_url = 'https://s3.amazonaws.com/text-datasets/imdb.npz'
    file_name = 'imdb.npz'

    file_path = download_dataset(file_url, file_name)

    # %%

    return file_path


# %%

if __name__ == '__main__':
    dataset_path = download_imdb('../../../')
