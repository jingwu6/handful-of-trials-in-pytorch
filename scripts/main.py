from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse
import pprint

# from .Controller import Controller
from dotmap import DotMap
from controllers.MPC import MPC
from src.mbrl.config import create_config
from src.MBExperiment import MBExperiment


def main(env, ctrl_type, ctrl_args, overrides, logdir):
    ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})
    cfg = create_config(env, ctrl_type, ctrl_args, overrides, logdir)
    #  print the structure of the class
    # cfg.pprint()

    # print('Here', cfg.ctrl_cfg.opt_cfg.get("cfg", {}))

    assert ctrl_type == 'MPC'

    cfg.exp_cfg.exp_cfg.policy = MPC(cfg.ctrl_cfg)

    # cfg.pprint()

    exp = MBExperiment(cfg.exp_cfg)
    # print("dic",exp.logdir)
    os.makedirs(exp.logdir)
    with open(os.path.join(exp.logdir, "config.txt"), "w") as f:
        f.write(pprint.pformat(cfg.toDict()))

    exp.run_experiment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument('-env', type=str,
    #                     default='halfcheetah',
    #                     help='Environment name: select from [cartpole, reacher, pusher, halfcheetah]')
    # parser.add_argument('-env', type=str,
    #                     default='reacher',
    #                     help='Environment name: select from [cartpole, reacher, pusher, halfcheetah]')
    parser.add_argument('-env', type=str,
                        default='cartpole',
                        help='Environment name: select from [cartpole, reacher, pusher, halfcheetah]')

    parser.add_argument('-ca', '--ctrl_arg', action='append', nargs=2,
                        default=[['model-type', 'PE'], ['prop-type', 'TSinf'], ['opt-type', 'CEM']])

    parser.add_argument('-o', '--override', action='append', nargs=2,
                        default=[])

    parser.add_argument('-logdir', type=str,
                        default='D:\model_based_meta\log',
                        help='Directory to which results will be logged (default: ./log)')
    args = parser.parse_args()

    main(args.env, "MPC", args.ctrl_arg, args.override, args.logdir)

#     print(args.env)
#     print(args.ctrl_arg)
#     print(args.override)
#     print(args.logdir)
#     test = args.ctrl_arg
#     for (key, val) in test:
#         print(key, val)
#     ctrl_args = DotMap(**{key: val for (key, val) in test})
#     print(ctrl_args)
#     print(ctrl_args.get("model-type", "PE"), len(ctrl_args["model-type"]))
#     main(args.env, "MPC", args.ctrl_arg, args.override, args.logdir)
#
# print(os.path.dirname(os.path.realpath(__file__)))
