import pickle
import os
import torch
from torch import Tensor
import numpy as np
import time

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path

import logging

from sbi.inference import MCMCPosterior
from sbi.utils import mcmc_transform
from sbi.utils.metrics import c2st

from pathlib import Path

# Algorithm imports.
from sbi.inference import SNPE, SNLE
from gbi.GBI import GBInference
import gbi.utils.utils as gbi_utils

# Task imports.
from gbi.benchmark.tasks.uniform_1d.task import UniformNoise1D
from gbi.benchmark.tasks.two_moons.task import TwoMoonsGBI
from gbi.benchmark.tasks.linear_gaussian.task import LinearGaussian
from gbi.benchmark.tasks.gaussian_mixture.task import GaussianMixture
from gbi import distances


@hydra.main(version_base="1.1", config_path="config", config_name="run_inference")
def run_inference(cfg: DictConfig) -> None:    
    # Get directory of, and load trained inference algorithm.

    inference_folder = f'../../'
    inference = gbi_utils.pickle_load(inference_folder + 'inference.pickle')
    print(inference)

    # Get high-level task path.
    dir_path = get_original_cwd()
    full_path_prepend = f"{dir_path}/../tasks/{cfg.task.name}/"
    print(full_path_prepend)

    # inference_folder = full_path_prepend + "/trained_inference/"    

    # Get observation directory.
    observation_folder = full_path_prepend + "/xos/"    

    # Get ground-truth directory.
    print('-----')
    # print(os.listdir(f'../../../../../ground_truths/{cfg.task.name}/{cfg.gt_datetime}/')) # This is super fucking ugly.
    # gt_folder = f'../{cfg.gt_datetime}/{cfg.algorithm.name}/'
    gt_folder = f'../../../../../../ground_truths/{cfg.task.name}/{cfg.gt_datetime}/beta_{cfg.task.beta}/'
    print(gt_folder, os.listdir(gt_folder))

    # ### Define task and distance function.
    # distance_func = distances.mse_dist
    # if cfg.task.name == "linear_gaussian":
    #     Task = LinearGaussian
    # elif cfg.task.name == "two_moons":
    #     Task = TwoMoonsGBI
    # elif cfg.task.name == "uniform_1d":
    #     Task = UniformNoise1D
    # elif cfg.task.name == "gaussian_mixture":
    #     Task = GaussianMixture
    #     distance_func = distances.mmd_dist
    # else:
    #     raise NameError

    # ### Sample and simulate from task.
    # # Set seed
    # if cfg.seed is None:
    #     seed = int((time.time() % 1) * 1e7)
    # else:
    #     seed = cfg.seed
    # np.savetxt(full_path_prepend + "seed.txt", np.asarray([seed]))

    # # Sample and simulate.
    # _ = torch.manual_seed(seed)
    # _ = np.random.seed(seed=seed)
    # task = Task(seed=seed)
    
    # # Train inference algorithms.
    # task.dist_func_gbi = distance_func

    # # Get trained inference object.
    # inference = gbi_utils.pickle_load(inference_folder + 'GBI.pickle')
    # print(inference)

    # # Load all observations.
    



    # potential_fn = inference.get_potential(x_o=x_o, beta=cfg.task.beta)
    # theta_transform = mcmc_transform(task.prior)
    # posterior = MCMCPosterior(
    #     potential_fn,
    #     theta_transform=theta_transform,
    #     proposal=task.prior,
    #     method="slice_np_vectorized",
    #     thin=10,
    #     warmup_steps=50,
    #     num_chains=100,
    #     init_strategy="resample",
    # )
    # samples = posterior.sample((10_000,))

    # setup = f"{cfg.task.name}/2023_01_16__16_38_48/beta_{cfg.task.beta}/obs_{cfg.task.xo_index}"
    # with open(
    #     f"{dir_path}/../../results/benchmark/ground_truths/{setup}/rejection_samples.pkl",
    #     "rb",
    # ) as handle:
    #     gt_samples = pickle.load(handle)

    # assert isinstance(samples, Tensor)
    # c2st_val = c2st(samples, gt_samples)
    # log.info(f"c2st {c2st_val}")


if __name__ == "__main__":
    run_inference()


"""
pseudo code:
- for each task:
    - fix seed
    - sample theta, simulate x. [simulation_budget]
    - save training sims ??
    
    > training
    - GBI [network_params]
        - set task specific dist_func?
        - include x_os ??
    - NPE, NLE, ABC [params]
    - save trained networks ??

    > sampling [n_samples]
    - for each x_o
        - for each beta:
            - sample {GBI, NLE}
        - sample {NPE, ABC}
        - simulate
        - save test sims ??

    > evaluate
    - for each x_o, beta
        - c2st
        - posterior predictive distance
"""




