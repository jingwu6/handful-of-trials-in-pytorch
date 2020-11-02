import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import torch
from torch import nn as nn


def swish(x):
    return x * torch.sigmoid(x)


def truncated_normal(size, std):
    # We use TF to implement initialization function for neural network weight because:
    # 1. Pytorch doesn't support truncated normal
    # 2. This specific type of initialization is important for rapid progress early in training in cartpole

    # Do not allow tf to use gpu memory unnecessarily
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True

    sess = tf.Session(config=cfg)
    val = sess.run(tf.truncated_normal(shape=size, stddev=std))

    # Close the session and free resources
    sess.close()

    return torch.tensor(val, dtype=torch.float32)


def truncated_normal_torch(size, mean=0, std=1):
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    tensor = torch.zeros(size)

    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

    return torch.tensor(tensor, dtype=torch.float32)


# simple test of torch gaussian trucation function
# a = truncated_normal_torch([1, 3], std=1.0 / (2.0 * np.sqrt([3])))
# b = truncated_normal(size=[1, 3], std=1.0 / (2.0 * np.sqrt([3])))
# print(nn.Parameter(a), nn.Parameter(b))
# print(nn.Parameter(b))


def get_affine_params(ensemble_size, in_features, out_features):
    w = truncated_normal(size=(ensemble_size, in_features, out_features),
                         std=1.0 / (2.0 * np.sqrt(in_features)))
    w = nn.Parameter(w)
    b = nn.Parameter(torch.zeros(ensemble_size, 1, out_features, dtype=torch.float32))

    return w, b

# w,b=get_affine_params(5, 24, 200)

