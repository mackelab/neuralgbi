import pickle
import os
import torch
from torch import Tensor
import numpy as np
import time

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path
from sbi.inference import MCMCPosterior
from sbi.utils import mcmc_transform


# Algorithm imports.
from sbi.inference import SNPE, SNLE
from gbi.GBI import GBInference
import gbi.utils.utils as gbi_utils
from run_training import get_task_and_distance_func


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
        )
    posterior_samples = posterior.sample((n_samples,))
    return posterior_samples

def sample_NPE(inference, x_o, task, n_samples=10_000):
    return inference.build_posterior(prior=task.prior).set_default_x(x_o).sample((n_samples,))


@hydra.main(version_base="1.1", config_path="config", config_name="run_inference")
def run_inference(cfg: DictConfig) -> None:    
    # Get directory of, and load trained inference algorithm.
    inference_dir = f'../../../../'
    inference = gbi_utils.pickle_load(inference_dir + 'inference.pickle')    

    # Get high-level task path.
    dir_path = get_original_cwd()
    full_path_prepend = f"{dir_path}/../tasks/{cfg.task.name}/"

    # Get observation directory and load xo.
    observation_dir = full_path_prepend + "/xos/"
    obs_file = f'xo_{cfg.task.is_specified}_{cfg.task.is_known}.pkl'
    xos = gbi_utils.pickle_load(observation_dir + obs_file)

    # ### ACTUALLY I DON'T NEED THIS NOW, CAN LOAD GROUND TRUTH LATER IN COMPARISON SCRIPT.
    # # Get ground-truth posterior samples.
    # # This is super fucking ugly.
    # gt_dir = f'../../../../../../ground_truths/{cfg.task.name}/{cfg.gt_datetime}/beta_{cfg.task.beta}/'
    # xo_dir = gt_dir + f'/obs_{cfg.task.xo_index}_{cfg.task.is_specified}_{cfg.task.is_known}'
    # gt_posterior_samples = gbi_utils.pickle_load(xo_dir + '/rejection_samples.pkl')    

    # Set seed
    if cfg.seed is None:
        seed = int((time.time() % 1) * 1e7)
    else:
        seed = cfg.seed
    np.savetxt("seed.txt", np.asarray([seed]))
    _ = torch.manual_seed(seed)
    _ = np.random.seed(seed=seed)

    # Get task and distance function, only need for prior.
    Task, _ = get_task_and_distance_func(cfg)    
    task = Task(seed=seed)


    ### TO DO: INCLUDE NLE, NLE-distance, ABC algorithms here.
    ### Sample from inference algorithm and save
    if cfg.algorithm.name == 'NPE':
        posterior_samples = sample_NPE(inference, xos[cfg.task.xo_index], task)
    elif cfg.algorithm.name == 'GBI':
        posterior_samples = sample_GBI(inference, xos[cfg.task.xo_index], cfg.task.beta, task)

    gbi_utils.pickle_dump('posterior_samples.pkl', posterior_samples)




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




