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
    # with open(f"{dir_path}/../tasks/{cfg.task.name}/xo.pkl", "rb") as handle:
    #     simulated_x = pickle.load(handle)
    # x_o = simulated_x[cfg.task.xo_index]

    # with open(f"{dir_path}/../tasks/{cfg.task.name}/gt.pkl", "rb") as handle:
    #     gt = pickle.load(handle)
    # gt = gt[cfg.task.xo_index]

    with open(f"{dir_path}/../tasks/{cfg.task.name}/xos/xo_{cfg.task.is_specified}_{cfg.task.is_known}.pkl", "rb") as handle:
        simulated_x = pickle.load(handle)
    x_o = simulated_x[cfg.task.xo_index].unsqueeze(0)

    if cfg.task.is_specified == 'specified':
        # Specified GT theta exists, load
        with open(f"{dir_path}/../tasks/{cfg.task.name}/xos/theta_gt_{cfg.task.is_known}.pkl", "rb") as handle:
            gt = pickle.load(handle)        
        gt = gt[cfg.task.xo_index]
    else: 
        # Misspecified, no GT theta. Not sure what to do here.
        print('Misspecified, no GT theta. Not sure what to do here. lol')
        # Just use the x_o as the GT theta for now
        gt = x_o
        return

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


"""
### BUG
  File "run_gaussian_mixture.py", line 75, in run
    run_rejection(task, proposal=proposal, config=cfg.rejection)
  File "/home/rdgao/Documents/code/research/neuralgbi/gbi/benchmark/generate_gt/rejection.py", line 55, in run_rejection
    samples = posterior.sample((10_000,))
  File "/home/rdgao/Documents/code/research/neuralgbi/packages/sbi/sbi/inference/posteriors/rejection_posterior.py", line 150, in sample
    samples, _ = rejection_sample(
  File "/home/rdgao/Documents/code/research/neuralgbi/packages/sbi/sbi/samplers/rejection/rejection.py", line 73, in rejection_sample
    _, max_log_ratio = gradient_ascent(
  File "/home/rdgao/Documents/code/research/neuralgbi/packages/sbi/sbi/utils/sbiutils.py", line 786, in gradient_ascent
    init_probs = potential_fn(inits).detach()
  File "/home/rdgao/Documents/code/research/neuralgbi/packages/sbi/sbi/samplers/rejection/rejection.py", line 70, in potential_over_proposal
    return potential_fn(theta) - proposal.log_prob(theta)
  File "/home/rdgao/Documents/code/research/neuralgbi/gbi/benchmark/generate_gt/rejection.py", line 18, in __call__
    return self.task.potential(theta)
  File "/home/rdgao/Documents/code/research/neuralgbi/gbi/benchmark/tasks/gaussian_mixture/task.py", line 89, in potential
    term1 = -self.beta * self.distance_fn(t)
  File "/home/rdgao/Documents/code/research/neuralgbi/gbi/benchmark/tasks/gaussian_mixture/task.py", line 73, in distance_fn
    mmd_x = ground_truth_mmd(
  File "/home/rdgao/Documents/code/research/neuralgbi/gbi/utils/mmd.py", line 15, in ground_truth_mmd
    term1 = sample_based_mmd_marginal(
  File "/home/rdgao/Documents/code/research/neuralgbi/gbi/utils/mmd.py", line 52, in sample_based_mmd_marginal
    [
  File "/home/rdgao/Documents/code/research/neuralgbi/gbi/utils/mmd.py", line 53, in <listcomp>
    sample_based_mmd_term(x[:, d : d + 1], y[:, d : d + 1], scale=scale)
  File "/home/rdgao/Documents/code/research/neuralgbi/gbi/utils/mmd.py", line 63, in sample_based_mmd_term
    xo1 = x.repeat((num_y, 1))
RuntimeError: Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor
"""