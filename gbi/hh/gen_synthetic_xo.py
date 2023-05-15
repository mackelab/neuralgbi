import torch
import pickle
import numpy as np

import gbi.hh.utils as utils
from gbi.hh.HodgkinHuxley import HodgkinHuxley
from gbi.hh.HodgkinHuxleyStatsMoments import HodgkinHuxleyStatsMoments


def generate_synthetic_xo():
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

    # Generate 10 simulations for 100 thetas
    _ = torch.manual_seed(1)
    np.random.seed(1)

    nsims = 200
    theta_vals = prior.sample((nsims,))
    seeds = torch.randint(0, 100000, (nsims,))
    obs_stats_ls = []
    all_obs = []
    for theta_ind, theta_val in enumerate(theta_vals):
        x_val = sim.gen_single(theta_val, seed=seeds[theta_ind])
        all_obs.append(x_val)
        summstats = stats.calc([x_val])
        obs_stats_ls.append(summstats)

    obs_stats_ls = np.asarray(obs_stats_ls)
    obs_stats_ls = torch.as_tensor(obs_stats_ls, dtype=torch.float32)

    obs_stats_ls_condition = obs_stats_ls[:, :, 0] > 5.0
    obs_stats_ls = obs_stats_ls[obs_stats_ls_condition]

    all_obs_traces1 = []
    for o, c in zip(all_obs, obs_stats_ls_condition):
        if c:
            all_obs_traces1.append(o)

    obs_stats_ls_condition = obs_stats_ls[:, 0] < 40.0
    obs_stats_ls = obs_stats_ls[obs_stats_ls_condition].unsqueeze(1)
    all_obs_traces = []
    for o, c in zip(all_obs_traces1, obs_stats_ls_condition):
        if c:
            all_obs_traces.append(o)
    print("obs_stats_ls", obs_stats_ls)
    print("obs_stats_ls", obs_stats_ls.shape)
    print("all_obs_traces", len(all_obs_traces))

    with open("../../results/hh_allen/synthetic_xo/traces.pkl", "wb") as handle:
        pickle.dump(all_obs_traces[:10], handle)

    with open("../../results/hh_allen/synthetic_xo/xo.pkl", "wb") as handle:
        pickle.dump(obs_stats_ls[:10], handle)


if __name__ == "__main__":
    generate_synthetic_xo()
