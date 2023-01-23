import pickle
import os
import torch
import numpy as np
import time

import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd, to_absolute_path

import logging

from gbi.benchmark.tasks.two_moons.task import TwoMoonsGBI
from gbi.benchmark.generate_gt.mcmc import run_mcmc
from gbi.benchmark.generate_gt.flow import train_flow
from gbi.benchmark.generate_gt.rejection import run_rejection

log = logging.getLogger("run_benchmark_gt")


@hydra.main(version_base="1.1", config_path="config", config_name="run")
def run(cfg: DictConfig) -> None:

    dir_path = get_original_cwd()
    with open(f"{dir_path}/../tasks/{cfg.task.name}/xos/xo_{cfg.task.is_specified}_{cfg.task.is_known}.pkl", "rb") as handle:
        simulated_x = pickle.load(handle)
    x_o = simulated_x[cfg.task.xo_index].unsqueeze(0)

    # Define task.
    task = TwoMoonsGBI(x_o=x_o, beta=cfg.task.beta)

    if cfg.seed is None:
        seed = int((time.time() % 1) * 1e7)
    else:
        seed = cfg.seed
    np.savetxt("seed.txt", np.asarray([seed]))

    # Run ground truth suite.
    _ = torch.manual_seed(seed)
    _ = np.random.seed(seed=seed)
    run_rejection(task, proposal=task.prior, config=cfg.rejection)


if __name__ == "__main__":
    run()
