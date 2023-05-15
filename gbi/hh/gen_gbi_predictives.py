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

log = logging.getLogger("hh_gbi_pred")


def sample_gbi_all_xo(beta, observations, inference, prior):
    all_samples = []
    for xo in observations:
        xo = as_tensor(xo)
        potential_fn = inference.get_potential(x_o=xo, beta=beta)
        theta_transform = mcmc_transform(prior)

        posterior = MCMCPosterior(
            potential_fn,
            theta_transform=theta_transform,
            proposal=prior,
            method="slice_np_vectorized",
            thin=10,
            warmup_steps=50,
            num_chains=100,
            init_strategy="resample",
            frac_chains_to_finish=0.9,
        )
        posterior_samples = posterior.sample((1_000,), frac_chains_to_finish=0.9)
        all_samples.append(posterior_samples)
    return all_samples


@hydra.main(version_base="1.1", config_path="config", config_name="gbi_predictives")
def generate_predictives(cfg: DictConfig) -> None:
    with open(
        f"{get_original_cwd()}/../../results/hh_allen/gbi_inference/inference_gbi.pkl",
        "rb",
    ) as handle:
        inference_gbi = pickle.load(handle)

    n_xcorr = 0
    n_mom = 4
    n_summary = 7

    true_params, labels_params = utils.obs_params(reduced_model=False)

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

    prior = utils.prior(
        true_params=true_params,
        prior_uniform=True,
        prior_extent=True,
        prior_log=False,
        seed=0,
    )

    if cfg.observation == "allen":
        obs_stats_ls, all_obs = utils.load_all_allen()
    elif cfg.observation == "synthetic":
        path = f"{get_original_cwd()}/../../results/hh_allen/synthetic_xo/xo.pkl"
        with open(path, "rb") as handle:
            obs_stats_ls = pickle.load(handle)
    x_target_vals = torch.as_tensor(obs_stats_ls, dtype=torch.float32)

    # Generate 10 simulations for 100 thetas
    _ = torch.manual_seed(111)
    np.random.seed(111)

    theta_vals = prior.sample((100,))
    seeds = torch.randint(0, 100000, (100, 10))
    final_stats = []
    for theta_ind, theta_val in enumerate(theta_vals):
        all_summstats = []
        for iter_ in range(10):  # 10 sims per theta
            x_val = sim.gen_single(theta_val, seed=seeds[theta_ind, iter_])
            summstats = stats.calc([x_val])
            all_summstats.append(summstats)
        final_stats.append(all_summstats)

    final_stats = np.asarray(final_stats)
    final_stats = torch.as_tensor(final_stats, dtype=torch.float32)
    final_stats = final_stats.squeeze()

    with open(f"01_prior_theta.pkl", "wb") as handle:
        pickle.dump(theta_vals, handle)
    with open(f"01_prior_pred.pkl", "wb") as handle:
        pickle.dump(final_stats, handle)

    betas = cfg.beta

    _ = torch.manual_seed(222)

    all_posterior_gbi_samples = {}
    for beta in betas:
        samples_gbi = sample_gbi_all_xo(beta, x_target_vals, inference_gbi, prior)
        all_posterior_gbi_samples[f"beta_{int(beta)}"] = samples_gbi

    with open(f"02_posterior_samples_gbi_allen.pkl", "wb") as handle:
        pickle.dump(all_posterior_gbi_samples, handle)

    final_predictive_stats_gbi = {}
    final_predictive_traces_gbi = {}
    for key in all_posterior_gbi_samples.keys():
        posterior_samples_all_xos = all_posterior_gbi_samples[key]
        all_predictive_stats_gbi = []
        all_predictive_traces_gbi = []
        for posterior_samples in posterior_samples_all_xos:
            r = [
                sim.gen_single(theta.numpy(), seed=0)
                for theta in posterior_samples[:100]
            ]
            stats_ = stats.calc(r)
            all_predictive_stats_gbi.append(
                torch.as_tensor(stats_, dtype=torch.float32)
            )
            all_predictive_traces_gbi.append(r)

        final_predictive_stats_gbi[key] = all_predictive_stats_gbi
        final_predictive_traces_gbi[key] = all_predictive_traces_gbi

    with open(f"03_posterior_predictive_stats_gbi.pkl", "wb") as handle:
        pickle.dump(final_predictive_stats_gbi, handle)
    with open(f"03_posterior_predictive_traces_gbi.pkl", "wb") as handle:
        pickle.dump(final_predictive_traces_gbi, handle)


if __name__ == "__main__":
    generate_predictives()
