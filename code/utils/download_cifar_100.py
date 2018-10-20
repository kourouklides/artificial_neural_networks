"""

Utility to download the CIFAR-100 dataset

    Author: Ioannis Kourouklides, www.kourouklides.com
    License: https://github.com/kourouklides/artificial_neural_networks/blob/master/LICENSE

"""
#%%

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from download_dataset import download

file_url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
file_name = 'cifar-100-python.tar.gz'

download(file_url,file_name)
