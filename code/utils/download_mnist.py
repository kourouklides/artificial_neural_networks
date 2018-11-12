"""

Utility to download the MNIST dataset

    Author: Ioannis Kourouklides, www.kourouklides.com
    License: https://github.com/kourouklides/artificial_neural_networks/blob/master/LICENSE

"""
#%%

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from artificial_neural_networks.code.utils.download_dataset import download

file_url = 'https://s3.amazonaws.com/img-datasets/mnist.npz'
file_name = 'mnist.npz'

download(file_url,file_name)
