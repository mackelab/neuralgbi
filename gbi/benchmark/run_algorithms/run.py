# import pickle
# import os
# import torch
# from torch import Tensor
# import numpy as np
# import time

# import hydra
# from omegaconf import DictConfig
# from hydra.utils import get_original_cwd, to_absolute_path

# import logging

# from sbi.inference import MCMCPosterior
# from sbi.utils import mcmc_transform
# from sbi.utils.metrics import c2st

# from gbi.GBI import GBInference
# from gbi.benchmark.tasks.uniform_1d.task import UniformNoise1D
# from gbi.benchmark.tasks.two_moons.task import TwoMoonsGBI
# from gbi.benchmark.tasks.linear_gaussian.task import LinearGaussian
# from gbi.benchmark.tasks.gaussian_mixture.task import GaussianMixture
# from gbi import distances

# log = logging.getLogger("run_algo")


# @hydra.main(version_base="1.1", config_path="config", config_name="run")
# def run(cfg: DictConfig) -> None:
#     # Get high-level path.
#     dir_path = get_original_cwd()

#     full_path_prepend = f"{dir_path}/../tasks/{cfg.task.name}/"
#     ### WHERE SHOULD THINGS BE SAVED/LOADED?

#     ### Define task and distance function.
#     distance_func = distances.mse_dist
#     if cfg.task.name == "linear_gaussian":
#         Task = LinearGaussian
#     elif cfg.task.name == "two_moons":
#         Task = TwoMoonsGBI
#     elif cfg.task.name == "uniform_1d":
#         Task = UniformNoise1D
#     elif cfg.task.name == "gaussian_mixture":
#         Task = GaussianMixture
#         distance_func = distances.mmd_dist
#     else:
#         raise NameError

#     ### Sample and simulate from task.
#     # Set seed
#     if cfg.seed is None:
#         seed = int((time.time() % 1) * 1e7)
#     else:
#         seed = cfg.seed
#     np.savetxt(full_path_prepend + "seed.txt", np.asarray([seed]))

#     # Sample and simulate.
#     _ = torch.manual_seed(seed)
#     _ = np.random.seed(seed=seed)
#     task = Task(seed=seed)
#     theta = Task.prior.sample((cfg.training_simulation_budget,))
#     x = Task.simulate(theta)

#     # Save simulations.
#     ### SAVE?

#     # Train inference algorithms.
#     train_GBI()  # Should GBI get access to observations?
#     train_NPE()
#     train_NLE()
#     train_ABC()

#     ####### Training script should end here.

#     ### Inference.
#     # Get x_o
#     dir_path = get_original_cwd()
#     with open(f"{dir_path}/../tasks/{cfg.task.name}/xo.pkl", "rb") as handle:
#         simulated_x = pickle.load(handle)
#     x_o = simulated_x[cfg.task.xo_index].unsqueeze(0)

#     # # Define task.
#     # if cfg.task.name == "linear_gaussian":
#     #     task = LinearGaussian(x_o=x_o, beta=cfg.task.beta)
#     # else:
#     #     raise NameError

#     # Load inference object (and create posterior if necessary).

#     # Sample posterior.

#     # Simulate posterior predictives

#     # Compute metrics.
#     ## C2ST
#     ## Mean distance of posterior predictives.
#     # Something else that rewards broad coverage?

#     _ = torch.manual_seed(seed)
#     _ = np.random.seed(seed=seed)

#     # Generate simulations
#     theta = task.prior.sample((cfg.task.num_simulations,))  # type: ignore
#     x = task.simulate(theta)
#     x_target = x + torch.randn(x.shape) * torch.std(x, dim=0) * cfg.net.noise_level

#     # Run algorithm.
#     inference = GBInference(
#         prior=task.prior, distance_func=None, do_precompute_distances=True
#     )
#     inference = inference.append_simulations(theta, x, x_target=x_target)
#     inference.initialize_distance_estimator(
#         num_layers=cfg.net.num_layers,
#         num_hidden=cfg.net.num_hidden,
#         net_type=cfg.net.net_type,
#     )
#     inference.train(training_batch_size=cfg.net.training_batch_size)
#     potential_fn = inference.get_potential(x_o=x_o, beta=cfg.task.beta)
#     theta_transform = mcmc_transform(task.prior)
#     posterior = MCMCPosterior(
#         potential_fn,
#         theta_transform=theta_transform,
#         proposal=task.prior,
#         method="slice_np_vectorized",
#         thin=10,
#         warmup_steps=50,
#         num_chains=100,
#         init_strategy="resample",
#     )
#     samples = posterior.sample((10_000,))

#     setup = f"{cfg.task.name}/2023_01_16__16_38_48/beta_{cfg.task.beta}/obs_{cfg.task.xo_index}"
#     with open(
#         f"{dir_path}/../../results/benchmark/ground_truths/{setup}/rejection_samples.pkl",
#         "rb",
#     ) as handle:
#         gt_samples = pickle.load(handle)

#     assert isinstance(samples, Tensor)
#     c2st_val = c2st(samples, gt_samples)
#     log.info(f"c2st {c2st_val}")


# if __name__ == "__main__":
#     run()


# """
# pseudo code:
# - for each task:
#     - fix seed
#     - sample theta, simulate x. [simulation_budget]
#     - save training sims ??
    
#     > training
#     - GBI [network_params]
#         - set task specific dist_func?
#         - include x_os ??
#     - NPE, NLE, ABC [params]
#     - save trained networks ??

#     > sampling [n_samples]
#     - for each x_o
#         - for each beta:
#             - sample {GBI, NLE}
#         - sample {NPE, ABC}
#         - simulate
#         - save test sims ??

#     > evaluate
#     - for each x_o, beta
#         - c2st
#         - posterior predictive distance
# """
