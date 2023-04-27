import torch
from torch import nn
import numpy as np
import time

import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

import logging
from sbi.utils import get_nn_models
from sbi.neural_nets.embedding_nets import PermutationInvariantEmbedding, FCEmbedding

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


def train_NPE(theta, x, task, config, task_name):
    if task_name == "gaussian_mixture":
        trial_net = FCEmbedding(input_dim=2, output_dim=20)
        embedding_net = PermutationInvariantEmbedding(
            trial_net=trial_net, trial_net_output_dim=20
        )
    else:
        embedding_net = nn.Identity()

    if (config.sigmoid_theta) and (task_name != "linear_gaussian"):
        # Apply sigmoid on theta to keep into prior range.
        # Do not apply to linear_gaussian because prior is unbounded. Should probably just check.
        net = get_nn_models.posterior_nn(
            model=config.density_estimator,
            sigmoid_theta=True,
            prior=task.prior,
            embedding_net=embedding_net,
        )
    else:
        # Regular NPE
        net = get_nn_models.posterior_nn(
            model=config.density_estimator,
            sigmoid_theta=False,
            embedding_net=embedding_net,
        )
    inference = SNPE(prior=task.prior, density_estimator=net)

    density_estimator = inference.append_simulations(theta, x).train()
    return inference, density_estimator


def train_NLE(theta, x, task, config, task_name):
    if task_name == "gaussian_mixture":
        # NLE is trained on single-trial simulations. So we reshaped the multi-trial
        # simulator output into single-trial simulations.
        batch_size, trial_num, data_size = x.shape
        x = torch.reshape(x, (batch_size * trial_num, data_size))
        theta = torch.repeat_interleave(theta, trial_num, dim=0)
    inference = SNLE(prior=task.prior, density_estimator=config.density_estimator)
    density_estimator = inference.append_simulations(theta, x).train()
    return inference, density_estimator


def train_eGBI(theta, x, task, distance_func, config, task_name):
    if task_name == "gaussian_mixture":
        # NLE is trained on single-trial simulations. So we reshaped the multi-trial
        # simulator output into single-trial simulations.
        batch_size, trial_num, data_size = x.shape
        x = torch.reshape(x, (batch_size * trial_num, data_size))
        theta = torch.repeat_interleave(theta, trial_num, dim=0)
    inference = SNLE(prior=task.prior, density_estimator=config.density_estimator)
    density_estimator = inference.append_simulations(theta, x).train()
    eGBI = GBInferenceEmulator(
        emulator_net=density_estimator,
        prior=task.prior,
        distance_func=distance_func,
        n_emulator_samples=config.n_emulator_samples,
    )
    return eGBI, density_estimator


def train_GBI(theta, x, task, config, task_folder, task_name):
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
    if task_name == "gaussian_mixture":
        net_kwargs = {"trial_net_input_dim": 2, "trial_net_output_dim": 20}
    else:
        net_kwargs = {"trial_net_input_dim": None, "trial_net_output_dim": None}
    inference = GBInference(
        prior=task.prior,
        distance_func=task.dist_func_gbi,
        do_precompute_distances=config.do_precompute_distances,
    )
    inference = inference.append_simulations(theta, x, x_target)
    inference.initialize_distance_estimator(
        num_layers=config.num_layers,
        num_hidden=config.num_hidden,
        net_type=config.net_type,
        positive_constraint_fn=config.positive_constraint_fn,
        net_kwargs=net_kwargs,
    )
    distance_net = inference.train(
        training_batch_size=config.training_batch_size,
        max_n_epochs=config.max_epochs,
        validation_fraction=config.validation_fraction,
        n_train_per_theta=config.n_train_per_theta,
        n_val_per_theta=config.n_val_per_theta,
        stop_after_counter_reaches=config.n_epochs_convergence,
        print_every_n=config.print_every_n,
        plot_losses=True,
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
        inference, _ = train_GBI(
            theta, x, task, cfg.algorithm, task_folder, cfg.task.name
        )

    elif cfg.algorithm.name == "NPE":
        inference, _ = train_NPE(theta, x, task, cfg.algorithm, cfg.task.name)

    elif cfg.algorithm.name == "NLE":
        inference, _ = train_NLE(theta, x, task, cfg.algorithm, cfg.task.name)

    elif cfg.algorithm.name == "eGBI":
        inference, _ = train_eGBI(
            theta, x, task, distance_func, cfg.algorithm, cfg.task.name
        )

    elif cfg.algorithm.name == "ABC":
        inference = ABC().append_simulations(theta, x).set_dist_fn(distance_func)

    else:
        raise NameError

    # Save inference object
    gbi_utils.pickle_dump("inference.pickle", inference)
    return


if __name__ == "__main__":
    run_training()
