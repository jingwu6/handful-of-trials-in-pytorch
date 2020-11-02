from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from dotmap import DotMap

import gym
import pybullet_envs
import pybulletgym  # register PyBullet enviroments with open ai gym2

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from modeling.models import BNN
from src.mbrl.misc.DotmapUtils import get_required_argument


TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class CartpoleConfigModule:
    ENV_NAME = 'InvertedPendulumPyBulletEnv-v0'
    TASK_HORIZON = 200
    NTRAIN_ITERS = 1
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 25
    MODEL_IN, MODEL_OUT = 7, 5
    GP_NINDUCING_POINTS = 200

    # Create and move this tensor to GPU so that
    # we do not waste time moving it repeatedly to GPU later
    ee_sub = torch.tensor([0.0, 0.6], device=TORCH_DEVICE, dtype=torch.float)

    def __init__(self):
        self.ENV = gym.make(self.ENV_NAME)
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2000
            },
            "CEM": {
                "popsize": 400,
                "num_elites": 40,
                "max_iters": 5,
                "alpha": 0.1
            }
        }

    @staticmethod
    def obs_preproc(obs):
        if isinstance(obs, np.ndarray):
           return np.concatenate([np.sin(obs[:, 1:2]), np.cos(obs[:, 1:2]), obs[:, :1], obs[:, 2:]], axis=1)
        elif isinstance(obs, torch.Tensor):
            return torch.cat([
                obs[:, 1:2].sin(),
                obs[:, 1:2].cos(),
                obs[:, :1],
                obs[:, 2:]
            ], dim=1)

    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    @staticmethod
    def obs_cost_fn(obs):
        ee_pos = CartpoleConfigModule._get_ee_pos(obs)

        ee_pos -= CartpoleConfigModule.ee_sub

        ee_pos = ee_pos ** 2

        ee_pos = - ee_pos.sum(dim=1)

        return - (ee_pos / (0.6 ** 2)).exp()

    @staticmethod
    def ac_cost_fn(acs):
        return 0.01 * (acs ** 2).sum(dim=1)

    @staticmethod
    def _get_ee_pos(obs):
        x0, theta = obs[:, :1], obs[:, 1:2]

        return torch.cat([
            x0 - 0.6 * theta.sin(), -0.6 * theta.cos()
        ], dim=1)

    def nn_constructor(self, model_init_cfg):
        # print('inital cfg', model_init_cfg)
        ensemble_size = get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size")

        load_model = model_init_cfg.get("load_model", False)

        assert load_model is False, 'Has yet to support loading model'

        model = BNN(ensemble_size,
                        self.MODEL_IN, self.MODEL_OUT * 2).to(TORCH_DEVICE)
        # * 2 because we output both the mean and the variance

        model.optim = torch.optim.Adam(model.parameters(), lr=0.001)

        return model


CONFIG_MODULE = CartpoleConfigModule