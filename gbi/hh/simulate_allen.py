import gbi.hh.utils as utils
from gbi.hh.HodgkinHuxley import HodgkinHuxley
from gbi.hh.HodgkinHuxleyStatsMoments import HodgkinHuxleyStatsMoments
import numpy as np
import time
import pickle
import torch


seed = 1
n_xcorr = 0
n_mom = 4
n_summary = 7


true_params, labels_params = utils.obs_params(reduced_model=False)

I, t_on, t_off, dt = utils.syn_current()

# define prior
p = utils.prior(
    true_params=true_params,
    prior_uniform=True,
    prior_extent=True,
    prior_log=False,
    seed=seed,
)

seed = 1
n_xcorr = 0
n_mom = 4
n_summary = 7

# list of all Allen recordings
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

obs_stats = utils.allen_obs_stats(
    data=obs,
    ephys_cell=ephys_cell,
    sweep_number=sweep_number,
    n_xcorr=n_xcorr,
    n_mom=n_mom,
    n_summary=n_summary,
)


sim = HodgkinHuxley(
    I, dt, V0=obs["data"][0], reduced_model=False, cython=True, prior_log=False
)
stats = HodgkinHuxleyStatsMoments(
    t_on=t_on, t_off=t_off, n_xcorr=n_xcorr, n_mom=n_mom, n_summary=n_summary
)

n_runs = 1_000
print("Starting sims")

for run_ind in range(n_runs):
    n_sim = 1_000
    start_seed = n_sim * run_ind + 1
    end_seed = n_sim * (run_ind + 1) + 1
    seeds = np.arange(start_seed, end_seed, 1)
    _ = torch.manual_seed(start_seed)
    thetas = p.sample((n_sim,))

    start_time = time.time()
    r = [sim.gen_single(theta, seed=seed) for theta, seed in zip(thetas, seeds)]
    ss = stats.calc(r)
    print("time", time.time() - start_time)

    with open(f"allen_data/allen_theta_{run_ind}.pkl", "wb") as handle:
        pickle.dump(thetas, handle)

    with open(f"allen_data/allen_summstats_{run_ind}.pkl", "wb") as handle:
        pickle.dump(ss, handle)
