# import properties of latest version
from __future__ import division
from __future__ import print_function
# use absolute path instead of the current path
from __future__ import absolute_import

import numpy as np
import torch
from torch.autograd import Variable


class TensorStandardScaler:
    '''Helper class for automatically normalizing inputs into the network.
    '''

    def __init__(self, x_dim):
        '''Initializes a scaler.

        Arguments:
        x_dim (int): The dimensionality of the inputs into the scaler.
        Returns: None.
        '''

        # self.mu = Variable(torch.zeros([1, x_dim]), requires_grad=True)
        self.mu = Variable(torch.zeros([1, x_dim]))
        self.sigma = Variable(torch.ones([1, x_dim]))

        self.cached_mu, self.cached_sigma = np.zeros([0, x_dim]), np.ones([1, x_dim])

        def fit(self, data):
            """Runs two ops, one for assigning the mean of the data to the internal mean, and
            another for assigning the standard deviation of the data to the internal standard deviation.
            Arguments:
            data (np.ndarray): A numpy array containing the input
            Returns: None.
            """
            mu = np.mean(data, axis=0, keepdims=True)
            sigma = np.std(data, axis=0, keepdims=True)
            sigma[sigma < 1e-12] = 1.0

            self.mu = mu
            self.sigma = sigma
            self.fitted = True
            self.cache()
