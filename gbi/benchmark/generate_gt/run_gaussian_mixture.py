import pickle
import os
import torch
from torch._C import Value
from torch.distributions import MultivariateNormal
import numpy as np
import time

import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd, to_absolute_path

import logging

from gbi.benchmark.tasks.gaussian_mixture.task import GaussianMixture
from gbi.benchmark.generate_gt.mcmc import run_mcmc
from gbi.benchmark.generate_gt.flow import train_flow
from gbi.benchmark.generate_gt.rejection import run_rejection

log = logging.getLogger("run_benchmark_gt")


@hydra.main(version_base="1.1", config_path="config", config_name="run")
def run(cfg: DictConfig) -> None:

    dir_path = get_original_cwd()

    # TO DO: change to new folder structure
    with open(f"{dir_path}/../tasks/{cfg.task.name}/xo.pkl", "rb") as handle:
        simulated_x = pickle.load(handle)
    x_o = simulated_x[cfg.task.xo_index]

    with open(f"{dir_path}/../tasks/{cfg.task.name}/gt.pkl", "rb") as handle:
        gt = pickle.load(handle)
    gt = gt[cfg.task.xo_index]

    # Define task.
    task = GaussianMixture(x_o=x_o, beta=cfg.task.beta)

    if cfg.seed is None:
        seed = int((time.time() % 1) * 1e7)
    else:
        seed = cfg.seed
    np.savetxt("seed.txt", np.asarray([seed]))

    # Define proposal for rejection sampling.
    if task.beta == 2.0:
        var = torch.tensor(4.0) ** 2
    elif task.beta == 10.0:
        var = torch.tensor(2.0) ** 2
    elif task.beta == 50.0:
        var = torch.tensor(1.0) ** 2
    else:
        raise ValueError

    proposal = MultivariateNormal(gt, var * torch.eye(2))

    # Run ground truth suite.
    _ = torch.manual_seed(seed)
    _ = np.random.seed(seed=seed)
    run_rejection(task, proposal=proposal, config=cfg.rejection)


if __name__ == "__main__":
    run()
