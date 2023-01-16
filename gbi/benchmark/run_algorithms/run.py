import pickle
import os
import torch
import numpy as np
import time

import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd, to_absolute_path

import logging

from sbi.inference import MCMCPosterior
from sbi.utils import mcmc_transform
from sbi.utils.metrics import c2st

from gbi.GBI import GBInference
from gbi.benchmark.tasks.linear_gaussian.task import LinearGaussian

log = logging.getLogger("run_algo")


@hydra.main(version_base="1.1", config_path="config", config_name="run")
def run(cfg: DictConfig) -> None:

    dir_path = get_original_cwd()

    with open(f"{dir_path}/../tasks/{cfg.task.name}/xo.pkl", "rb") as handle:
        simulated_x = pickle.load(handle)
    x_o = simulated_x[cfg.task.xo_index].unsqueeze(0)

    # Define task.
    if cfg.task.name == "linear_gaussian":
        task = LinearGaussian(x_o=x_o, beta=cfg.task.beta)
    else:
        raise NameError

    if cfg.seed is None:
        seed = int((time.time() % 1) * 1e7)
    else:
        seed = cfg.seed
    np.savetxt("seed.txt", np.asarray([seed]))

    _ = torch.manual_seed(seed)
    _ = np.random.seed(seed=seed)

    # Generate simulations
    theta = task.prior.sample((cfg.task.num_simulations,))  # type: ignore
    x = task.simulate(theta)
    x_target = x + torch.randn(x.shape) * torch.std(x, dim=0) * cfg.net.noise_level

    # Run algorithm.
    inference = GBInference(
        prior=task.prior, distance_func=None, do_precompute_distances=True
    )
    inference.initialize_distance_estimator(
        num_layers=cfg.net.num_layers,
        num_hidden=cfg.net.num_hidden,
        net_type=cfg.net.net_type,
    )
    _ = inference.append_simulations(theta, x, x_target=x_target).train(
        training_batch_size=cfg.net.training_batch_size
    )
    # potential_fn = inference.get_potential()
    # theta_transform = mcmc_transform(task.prior)
    # posterior = MCMCPosterior(potential_fn, theta_transform=theta_transform, proposal=task.prior,
    #     method="slice_np_vectorized",
    #     thin=10,
    #     warmup_steps=50,
    #     num_chains=100,
    #     init_strategy="resample",)

    # samples = posterior.sample((10_000,))

    # with open("", "rb") as handle:
    #     gt_samples = pickle.load(handle)
    # c2st_val = c2st(samples, gt_samples)


if __name__ == "__main__":
    run()
