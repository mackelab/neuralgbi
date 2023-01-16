import pickle
from typing import Optional
import os

import torch
from gbi.tasks.linear_gaussian.task import Task


def generate_xo():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    _ = torch.manual_seed(0)
    task = Task()
    theta_gt = task.prior.sample((10,))
    simulated_x = task.linear_gaussian(theta_gt)
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
