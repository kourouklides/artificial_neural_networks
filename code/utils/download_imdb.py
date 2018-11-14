"""

Utility to download the IMDB dataset

    Author: Ioannis Kourouklides, www.kourouklides.com
    License: https://github.com/kourouklides/artificial_neural_networks/blob/master/LICENSE

"""
#%%

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

#%% 
def download_imdb(new_dir = os.getcwd()):
    
    os.chdir(new_dir)
    
    from artificial_neural_networks.code.utils.data_utils import download_dataset
    
    file_url = 'https://s3.amazonaws.com/text-datasets/imdb.npz'
    file_name = 'imdb.npz'
    
    file_path = download_dataset(file_url, file_name)
    
    return file_path

if __name__ == '__main__':
    file_path = download_imdb('../../../')

