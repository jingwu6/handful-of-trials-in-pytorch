from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from dotmap import DotMap

import gym
import pybullet_envs

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from modeling.models import BNN
from src.mbrl.misc.DotmapUtils import get_required_argument


TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class HalfCheetahConfigModule:
    ENV_NAME = "HalfCheetahBulletEnv-v0"
    TASK_HORIZON = 1000
    NTRAIN_ITERS = 300
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 30
    # MODEL_IN, MODEL_OUT = 24, 18
    MODEL_IN, MODEL_OUT = 32, 26

    GP_NINDUCING_POINTS = 300

    def __init__(self):
        self.ENV = gym.make(self.ENV_NAME)
        # cfg = tf.ConfigProto()
        # cfg.gpu_options.allow_growth = True
        # self.SESS = tf.Session(config=cfg)
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2500
            },
            "CEM": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
            }
        }

    @staticmethod
    def obs_preproc(obs):
        if isinstance(obs, np.ndarray):
            return np.concatenate([obs[:, 1:2], np.sin(obs[:, 2:3]), np.cos(obs[:, 2:3]), obs[:, 3:]], axis=1)
        else:
            return torch.cat([obs[:, 1:2], torch.sin(obs[:, 2:3]), torch.cos(obs[:, 2:3]), obs[:, 3:]], axis=1)

    @staticmethod
    def obs_postproc(obs, pred):
        if isinstance(obs, np.ndarray):
            return np.concatenate([pred[:, :1], obs[:, 1:] + pred[:, 1:]], axis=1)
        else:
            return torch.cat([pred[:, :1], obs[:, 1:] + pred[:, 1:]], axis=1)

    @staticmethod
    def targ_proc(obs, next_obs):
        return np.concatenate([next_obs[:, :1], next_obs[:, 1:] - obs[:, 1:]], axis=1)

    @staticmethod
    def obs_cost_fn(obs):
        return -obs[:, 0]

    @staticmethod
    def ac_cost_fn(acs):
        return 0.1 * (acs ** 2).sum(dim=1)


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

    # def gp_constructor(self, model_init_cfg):
    #     model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
    #         name="model",
    #         kernel_class=get_required_argument(model_init_cfg, "kernel_class", "Must provide kernel class"),
    #         kernel_args=model_init_cfg.get("kernel_args", {}),
    #         num_inducing_points=get_required_argument(
    #             model_init_cfg, "num_inducing_points", "Must provide number of inducing points."
    #         ),
    #         sess=self.SESS
    #     ))
    #     return model


CONFIG_MODULE = HalfCheetahConfigModule



