import pickle
import os
import torch
import numpy as np

import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd, to_absolute_path

import logging

from gbi.tasks.linear_gaussian.task import Task
from gbi.generate_gt.mcmc import run_mcmc
from gbi.generate_gt.flow import train_flow
from gbi.generate_gt.rejection import run_rejection

log = logging.getLogger("run_benchmark_gt")


@hydra.main(config_path="config", config_name="run")
def run(cfg: DictConfig) -> None:

    dir_path = get_original_cwd()

    with open(f"{dir_path}/../tasks/{cfg.task.name}/xo.pkl", "rb") as handle:
        simulated_x = pickle.load(handle)
    x_o = simulated_x[cfg.task.xo_index].unsqueeze(0)

    # Define task.
    task = Task(x_o=x_o)

    # Run ground truth suite.
    _ = torch.manual_seed(cfg.seed)
    _ = np.random.seed(seed=cfg.seed)
    run_mcmc(task)
    train_flow(**cfg.flow)
    run_rejection(task)


if __name__ == "__main__":
    run()
