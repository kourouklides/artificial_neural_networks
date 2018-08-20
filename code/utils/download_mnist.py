"""

Utility to download the MNIST dataset

    Author: Ioannis Kourouklides, www.kourouklides.com

"""
#%%

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves.urllib.request import urlretrieve

mnist_url = 'https://s3.amazonaws.com/img-datasets/mnist.npz'
mnist_path = '../../datasets/mnist.npz'

urlretrieve(mnist_url, mnist_path)

print("MNIST dataset was downloaded succesfully.")
