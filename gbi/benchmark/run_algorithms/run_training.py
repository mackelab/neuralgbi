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

log = logging.getLogger("run_algo")


def train_NPE(theta, x, task, config, task_folder):
    print("-------")
    print("Training NPE...")
    inference = SNPE(prior=task.prior, density_estimator=config.density_estimator)
    density_estimator = inference.append_simulations(theta, x).train()
    gbi_utils.pickle_dump(task_folder + "/trained_inference/NPE.pickle", inference)
    print("-------")
    return inference, density_estimator


def train_NLE(theta, x, task, config, task_folder):
    print("-------")
    print("Training NLE...")
    inference = SNLE(prior=task.prior, density_estimator=config.density_estimator)
    density_estimator = inference.append_simulations(theta, x).train()
    gbi_utils.pickle_dump(task_folder + "/trained_inference/NLE.pickle", inference)
    print("-------")
    return inference, density_estimator


def train_GBI(theta, x, task, config, task_folder):
    print("-------")
    print("Training GBI...")
    # Augment data with noise.
    x_aug = x[torch.randint(x.shape[0], size=(config.n_augmented_x,))]
    x_aug = x_aug + torch.randn(x_aug.shape) * x.std(dim=0) * config.noise_level
    x_target = gbi_utils.concatenate_xs(x, x_aug)

    # Append observations.
    xo_specified_known = gbi_utils.pickle_load(
        task_folder + "/xos/xo_specified_known.pkl"
    )
    xo_misspecified_known = gbi_utils.pickle_load(
        task_folder + "/xos/xo_misspecified_known.pkl"
    )
    x_obs = gbi_utils.concatenate_xs(xo_specified_known, xo_misspecified_known)

    # Put all together.
    x_target = gbi_utils.concatenate_xs(x_target, x_obs)

    # Initialize and train.
    inference = GBInference(prior=task.prior, distance_func=task.dist_func_gbi)
    inference = inference.append_simulations(theta, x, x_target)
    inference.initialize_distance_estimator(
        num_layers=config.num_layers,
        num_hidden=config.num_hidden,
        net_type=config.net_type,
        positive_constraint_fn=config.positive_constraint_fn,
    )
    distance_net = inference.train(
        training_batch_size=config.training_batch_size,
        max_n_epochs=config.max_epochs,
        print_every_n=config.print_every_n,
        plot_losses=False,
    )    
    gbi_utils.pickle_dump(task_folder + "/trained_inference/GBI.pickle", inference)
    print("-------")
    return inference, distance_net


@hydra.main(version_base="1.1", config_path="config", config_name="run")
def run_training(cfg: DictConfig) -> None:
    # Get high-level path.
    dir_path = get_original_cwd()
    full_path_prepend = f"{dir_path}/../tasks/{cfg.task.name}/"
    print(full_path_prepend)

    # Save to inference directory.
    inference_folder = full_path_prepend + "/trained_inference/"
    Path(inference_folder).mkdir(parents=True, exist_ok=True)

    ### WHERE SHOULD THINGS BE SAVED/LOADED?

    ### Define task and distance function.
    distance_func = distances.mse_dist
    if cfg.task.name == "linear_gaussian":
        Task = LinearGaussian
    elif cfg.task.name == "two_moons":
        Task = TwoMoonsGBI
    elif cfg.task.name == "uniform_1d":
        Task = UniformNoise1D
    elif cfg.task.name == "gaussian_mixture":
        Task = GaussianMixture
        distance_func = distances.mmd_dist
    else:
        raise NameError

    ### Sample and simulate from task.
    # Set seed
    if cfg.seed is None:
        seed = int((time.time() % 1) * 1e7)
    else:
        seed = cfg.seed
    np.savetxt(full_path_prepend + "seed.txt", np.asarray([seed]))

    # Sample and simulate.
    _ = torch.manual_seed(seed)
    _ = np.random.seed(seed=seed)
    task = Task(seed=seed)
    theta = task.prior.sample((cfg.training.training_simulation_budget,))
    x = task.simulate(theta)

    # Save simulations.
    ### TO DO

    # Train inference algorithms.
    task.dist_func_gbi = distance_func
    
    _, _ = train_NPE(theta, x, task, cfg.algorithm.NPE, full_path_prepend)
    _, _ = train_NLE(theta, x, task, cfg.algorithm.NLE, full_path_prepend)
    _, _ = train_GBI(theta, x, task, cfg.algorithm.GBI, full_path_prepend)

    


if __name__ == "__main__":
    run_training()
