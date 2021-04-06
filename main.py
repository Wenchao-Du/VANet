import os
from models.Solver import Solver
from torch.backends import cudnn
from congfig import config
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('--weights', type=str, default=None)
param = parse.parse_args()


def main(param):
    cudnn.benchmark = True
    if param.weights is not None:
        config.TRAIN.weights = param.weights
        print(config.TRAIN.weights)

    solver = Solver(config)
    if not os.path.exists(config.TRAIN.model_path):
        os.makedirs(config.TRAIN.model_path)
    if not os.path.exists(config.TRAIN.sample_path):
        os.makedirs(config.TRAIN.sample_path)

    if config.TRAIN.istrain:
        solver.train()


if __name__ == '__main__':
    main(param)
