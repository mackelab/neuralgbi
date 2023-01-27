import pickle
from typing import Optional
import os

import torch
from gbi.benchmark.tasks.gaussian_mixture.task import GaussianMixture


def generate_xo():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    _ = torch.manual_seed(0)
    task = GaussianMixture()
    theta_gt = task.prior.sample((10,))
    simulated_x = task.simulate(theta_gt)
    with open(f"{dir_path}/xo.pkl", "wb") as handle:
        pickle.dump(simulated_x, handle)
    with open(f"{dir_path}/gt.pkl", "wb") as handle:
        pickle.dump(theta_gt, handle)


def return_xo(index: Optional[int] = None):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    with open(f"{dir_path}/xo.pkl", "rb") as handle:
        simulated_x = pickle.load(handle)
    if index is not None:
        return simulated_x[index].unsqueeze(0)
    else:
        return simulated_x


def return_gt(index: Optional[int] = None):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    with open(f"{dir_path}/gt.pkl", "rb") as handle:
        simulated_x = pickle.load(handle)
    if index is not None:
        return simulated_x[index].unsqueeze(0)
    else:
        return simulated_x


if __name__ == "__main__":
    generate_xo()
