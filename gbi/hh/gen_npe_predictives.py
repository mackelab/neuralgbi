import pickle
import numpy as np
import torch
from torch import zeros, ones, eye, as_tensor
from sbi.inference import MCMCPosterior
from sbi.utils import mcmc_transform

import gbi.hh.utils as utils
from gbi.hh.HodgkinHuxley import HodgkinHuxley
from gbi.hh.HodgkinHuxleyStatsMoments import HodgkinHuxleyStatsMoments
from gbi.GBI import GBInference
from gbi.hh.train_gbi import MaeDistZ

from hydra.utils import get_original_cwd
import hydra
from omegaconf import DictConfig
import logging

log = logging.getLogger("hh_npe_pred")


@hydra.main(version_base="1.1", config_path="config", config_name="npe_predictives")
def generate_predictives(cfg: DictConfig) -> None:
    if cfg.nsims == 100_000:
        with open(
            f"{get_original_cwd()}/../../results/hh_allen/npe_inference_m/0_nsims_100000/inference_npe.pkl",
            "rb",
        ) as handle:
            inference_npe = pickle.load(handle)
    elif cfg.nsims == 1_000_000:
        with open(
            f"{get_original_cwd()}/../../results/hh_allen/npe_inference_m/1_nsims_1000000/inference_npe.pkl",
            "rb",
        ) as handle:
            inference_npe = pickle.load(handle)
    else:
        raise NameError

    n_xcorr = 0
    n_mom = 4
    n_summary = 7

    list_cells_AllenDB = [518290966, 57, 0.0234 / 126]

    ephys_cell = list_cells_AllenDB[0]
    sweep_number = list_cells_AllenDB[1]
    A_soma = list_cells_AllenDB[2]
    junction_potential = -14

    obs = utils.allen_obs_data(
        ephys_cell=ephys_cell, sweep_number=sweep_number, A_soma=A_soma
    )
    obs["data"] = obs["data"] + junction_potential
    I = obs["I"]
    dt = obs["dt"]
    t_on = obs["t_on"]
    t_off = obs["t_off"]

    sim = HodgkinHuxley(
        I, dt, V0=obs["data"][0], reduced_model=False, cython=True, prior_log=False
    )
    stats = HodgkinHuxleyStatsMoments(
        t_on=t_on, t_off=t_off, n_xcorr=n_xcorr, n_mom=n_mom, n_summary=n_summary
    )

    if cfg.observation == "allen":
        obs_stats_ls, all_obs = utils.load_all_allen()
    elif cfg.observation == "synthetic":
        path = f"{get_original_cwd()}/../../results/hh_allen/synthetic_xo/xo.pkl"
        with open(path, "rb") as handle:
            obs_stats_ls = pickle.load(handle)
    else:
        raise NameError
    x_target_vals = torch.as_tensor(obs_stats_ls, dtype=torch.float32)

    # Generate 10 simulations for 100 thetas
    _ = torch.manual_seed(111)
    np.random.seed(111)

    npe_all_samples = []
    for xo in obs_stats_ls:
        xo = as_tensor(xo)
        posterior = inference_npe.build_posterior()
        posterior_samples = posterior.sample((1_000,), x=xo, show_progress_bars=False)
        npe_all_samples.append(posterior_samples)

    _ = torch.manual_seed(3333)

    all_predictive_traces = []
    all_predictive_stats = []
    for posterior_samples in npe_all_samples:
        r = [sim.gen_single(theta.numpy(), seed=0) for theta in posterior_samples[:100]]
        all_predictive_traces.append(r)
        stats_ = stats.calc(r)
        all_predictive_stats.append(torch.as_tensor(stats_, dtype=torch.float32))

    with open("04_npe_theta.pkl", "wb") as handle:
        pickle.dump(npe_all_samples, handle)
    with open("04_npe_stats.pkl", "wb") as handle:
        pickle.dump(all_predictive_stats, handle)
    with open("04_npe_traces.pkl", "wb") as handle:
        pickle.dump(all_predictive_traces, handle)


if __name__ == "__main__":
    generate_predictives()
