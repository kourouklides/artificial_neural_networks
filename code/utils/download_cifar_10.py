"""

Utility to download the CIFAR-10 dataset

    Author: Ioannis Kourouklides, www.kourouklides.com
    License: https://github.com/kourouklides/artificial_neural_networks/blob/master/LICENSE

"""
#%%

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

initial_dir = os.getcwd()
os.chdir( '../../../' )
from artificial_neural_networks.code.utils.download_dataset import download_dataset
os.chdir(initial_dir)

#%% 
def download_cifar_10():
    file_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    file_name = 'cifar-10-python.tar.gz'
    
    file_path = download_dataset(file_url, file_name)
    
    return file_path

if __name__ == '__main__':
    file_path = download_cifar_10()

