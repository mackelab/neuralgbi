import torch
import numpy as np
import time

import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from sbi.inference import MCMCPosterior
from sbi.utils import mcmc_transform
import gbi.utils.utils as gbi_utils
from run_training import get_task_and_distance_func
import time
import logging

log = logging.getLogger("inference")


def sample_GBI(inference, x_o, beta, task, n_samples=10_000):
    potential_fn = inference.get_potential(x_o=x_o, beta=beta)
    theta_transform = mcmc_transform(task.prior)
    posterior = MCMCPosterior(
        potential_fn,
        theta_transform=theta_transform,
        proposal=task.prior,
        method="slice_np_vectorized",
        thin=10,
        warmup_steps=50,
        num_chains=100,
        init_strategy="resample",
        frac_chains_to_finish=0.9,
    )
    posterior_samples = posterior.sample((n_samples,))
    assert len(posterior_samples) == n_samples, "number of samples does not match."
    return posterior_samples


def sample_eGBI(
    inference,
    x_o,
    beta,
    task,
    n_samples=10_000,    
):
    potential_fn = inference.get_potential(x_o, beta)
    theta_transform = mcmc_transform(task.prior)
    posterior = MCMCPosterior(
        potential_fn,
        theta_transform=theta_transform,
        proposal=task.prior,
        method="slice_np_vectorized",
        thin=10,
        warmup_steps=50,
        num_chains=100,
        init_strategy="resample",
        frac_chains_to_finish=0.9,
    )
    posterior_samples = posterior.sample((n_samples,))
    assert len(posterior_samples) == n_samples, "number of samples does not match."
    return posterior_samples


def sample_NPE(inference, x_o, task, n_samples=10_000):
    return (
        inference.build_posterior(prior=task.prior)
        .set_default_x(x_o)
        .sample((n_samples,))
    )


def sample_NLE(inference, x_o, task, n_samples=10_000):
    posterior = inference.build_posterior(
        prior=task.prior,
        mcmc_method="slice_np_vectorized",
        mcmc_parameters={"num_chains": 100, "frac_chains_to_finish": 0.9},
    ).set_default_x(x_o)
    posterior_samples = posterior.sample((n_samples,))
    assert len(posterior_samples) == n_samples, "number of samples does not match."
    return posterior_samples


def sample_ABC(inference, x_o, beta, n_abc_samples):
    inference = inference.set_default_x(x_o)
    num_accepted = 0
    while num_accepted < n_abc_samples:
        posterior_samples = inference.sample(beta)
        num_accepted = len(posterior_samples)
        log.info(f"beta: {beta}, num_accepted: {num_accepted}")
        beta /= 1.1  # Reduce beta in order to make the posterior artificially broader.
    return posterior_samples[:n_abc_samples]


@hydra.main(version_base="1.1", config_path="config", config_name="run_inference")
def run_inference(cfg: DictConfig) -> None:
    # Get directory of, and load trained inference algorithm.
    inference_dir = f"../../../../"
    inference = gbi_utils.pickle_load(inference_dir + "inference.pickle")

    # Get high-level task path.
    dir_path = get_original_cwd()
    full_path_prepend = f"{dir_path}/../tasks/{cfg.task.name}/"

    # Get observation directory and load xo.
    observation_dir = full_path_prepend + "/xos/"
    obs_file = f"xo_{cfg.task.is_specified}_{cfg.task.is_known}.pkl"
    xos = gbi_utils.pickle_load(observation_dir + obs_file)

    # Set seed
    if cfg.seed is None:
        seed = int((time.time() % 1) * 1e7)
    else:
        seed = cfg.seed
    np.savetxt("seed.txt", np.asarray([seed]))
    _ = torch.manual_seed(seed)
    _ = np.random.seed(seed=seed)

    # Get task and distance function.
    Task, distance_function = get_task_and_distance_func(cfg)
    task = Task(seed=seed)

    ### Sample from inference algorithm and save
    n_samples = cfg.n_samples
    x_o = xos[cfg.task.xo_index]

    if cfg.task.name == "gaussian_mixture" and cfg.algorithm.name in ["NPE", "GBI"]:
        # Manually pad batch dimension for gaussian mixture, otherwise fails input check.
        # But not sure why it doesn't for the other algorithms...
        x_o = x_o[None, :, :]

    time_start = time.time()

    if cfg.algorithm.name == "NPE":
        posterior_samples = sample_NPE(inference, x_o, task, n_samples)

    elif cfg.algorithm.name == "NLE":
        posterior_samples = sample_NLE(inference, x_o, task, n_samples)

    elif cfg.algorithm.name == "ABC":
        posterior_samples = sample_ABC(
            inference, x_o, cfg.task.beta, cfg.algorithm.n_abc_samples
        )

    elif cfg.algorithm.name == "GBI":
        posterior_samples = sample_GBI(inference, x_o, cfg.task.beta, task, n_samples)

    elif cfg.algorithm.name == "eGBI":
        posterior_samples = sample_eGBI(
            inference,            
            x_o,
            cfg.task.beta,
            task,
            n_samples,            
        )

    log.info(f"Inference time taken: {time.time() - time_start}")
    gbi_utils.pickle_dump("posterior_samples.pkl", posterior_samples)


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
            - sample {GBI, emulatorGBI}
        - sample {NPE, ABC}
        - simulate
        - save test sims ??

    > evaluate
    - for each x_o, beta
        - c2st
        - posterior predictive distance
"""
