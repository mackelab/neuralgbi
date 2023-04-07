import torch
import numpy as np
import time

import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

import logging
from sbi.utils import get_nn_models

# Algorithm imports.
from sbi.inference import SNPE, SNLE
from gbi.GBI import GBInference, GBInferenceEmulator
import gbi.utils.utils as gbi_utils
from gbi.ABC import ABC

# Task imports.
from gbi.benchmark.tasks.uniform_1d.task import UniformNoise1D
from gbi.benchmark.tasks.two_moons.task import TwoMoonsGBI
from gbi.benchmark.tasks.linear_gaussian.task import LinearGaussian
from gbi.benchmark.tasks.gaussian_mixture.task import GaussianMixture
from gbi import distances

log = logging.getLogger("run_algo")


def get_task_and_distance_func(cfg):
    ### Define task and distance function.
    if cfg.task.name == "linear_gaussian":
        Task = LinearGaussian
    elif cfg.task.name == "two_moons":
        Task = TwoMoonsGBI
    elif cfg.task.name == "uniform_1d":
        Task = UniformNoise1D
    elif cfg.task.name == "gaussian_mixture":
        Task = GaussianMixture        
    else:
        raise NameError

    # Provide appropriate distance function
    if cfg.task.name == "gaussian_mixture":
        distance_func = distances.mmd_dist
    else:
        distance_func = distances.mse_dist

    return Task, distance_func

def train_NPE(theta, x, task, config):        
    if config.sigmoid_theta:
        # Apply sigmoid on theta to keep into prior range.
        net = get_nn_models.posterior_nn(model=config.density_estimator, sigmoid_theta=True, prior=task.prior)
        inference = SNPE(prior=task.prior, density_estimator=net)
    else:
        # Regular NPE
        inference = SNPE(prior=task.prior, density_estimator=config.density_estimator)    
    
    density_estimator = inference.append_simulations(theta, x).train()
    return inference, density_estimator


def train_NLE(theta, x, task, config):    
    inference = SNLE(prior=task.prior, density_estimator=config.density_estimator)
    density_estimator = inference.append_simulations(theta, x).train()
    return inference, density_estimator

def train_eGBI(theta, x, task, distance_func, config):
    inference = SNLE(prior=task.prior, density_estimator=config.density_estimator)
    density_estimator = inference.append_simulations(theta, x).train()
    eGBI = GBInferenceEmulator(emulator_net=density_estimator, prior=task.prior, distance_func=distance_func, n_emulator_samples=config.n_emulator_samples)
    return eGBI, density_estimator

def train_GBI(theta, x, task, config, task_folder):    
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
    return inference, distance_net


@hydra.main(version_base="1.1", config_path="config", config_name="run_training")
def run_training(cfg: DictConfig) -> None:
    # Load task and distance function.
    Task, distance_func = get_task_and_distance_func(cfg)

    ### Sample and simulate from task.
    # Set seed
    if cfg.seed is None:
        seed = int((time.time() % 1) * 1e7)
    else:
        seed = cfg.seed
    np.savetxt("seed.txt", np.asarray([seed]))

    # Sample and simulate.
    _ = torch.manual_seed(seed)
    _ = np.random.seed(seed=seed)
    task = Task(seed=seed)
    theta = task.prior.sample((cfg.task.num_simulations,))
    x = task.simulate(theta)

    print("----------------------")
    print(f"Training: {cfg.algorithm.name}...")
    if cfg.algorithm.name == "GBI":      
        # Get high-level path.
        dir_path = get_original_cwd()    
        task_folder = f"{dir_path}/../tasks/{cfg.task.name}/"        
        task.dist_func_gbi = distance_func
        inference, _ = train_GBI(theta, x, task, cfg.algorithm, task_folder)
    
    elif cfg.algorithm.name == "NPE":
        inference, _ = train_NPE(theta, x, task, cfg.algorithm)        
    
    elif cfg.algorithm.name == "NLE":
        inference, _ = train_NLE(theta, x, task, cfg.algorithm)

    elif cfg.algorithm.name == "eGBI":
        inference, _ = train_eGBI(theta, x, task, distance_func, cfg.algorithm)
    
    elif cfg.algorithm.name == "ABC":
        inference = ABC().append_simulations(theta, x).set_dist_fn(distance_func)
    
    else:
        raise NameError
    
    # Save inference object
    gbi_utils.pickle_dump("inference.pickle", inference)
    return

if __name__ == "__main__":
    run_training()
