import io
from sbi.utils import BoxUniform
from torch.distributions import MultivariateNormal
import inspect
import numpy as np
import os
import pickle
import gbi.hh.HodgkinHuxley as hh
from gbi.hh.HodgkinHuxleyStatsMoments import HodgkinHuxleyStatsMoments
from torch import zeros, ones, eye, as_tensor



def load_all_allen():
    seed = 1
    n_xcorr = 0
    n_mom = 4
    n_summary = 7

    # list of all Allen recordings
    list_cells_AllenDB = [
        [518290966, 57, 0.0234 / 126],
        [509881736, 39, 0.0153 / 184],
        [566517779, 46, 0.0195 / 198],
        [567399060, 38, 0.0259 / 161],
        [569469018, 44, 0.033 / 403],
        [532571720, 42, 0.0139 / 127],
        [555060623, 34, 0.0294 / 320],
        [534524026, 29, 0.027 / 209],
        [532355382, 33, 0.0199 / 230],
        [526950199, 37, 0.0186 / 218],
    ]

    n_cells = len(list_cells_AllenDB)

    # define model, summary statistics and generator
    obs_stats_ls = []
    m_ls = []
    s_ls = []
    g_ls = []
    all_obs = []
    for cell_num in range(n_cells):
        ephys_cell = list_cells_AllenDB[cell_num][0]
        sweep_number = list_cells_AllenDB[cell_num][1]
        A_soma = list_cells_AllenDB[cell_num][2]
        junction_potential = -14

        obs = allen_obs_data(
            ephys_cell=ephys_cell, sweep_number=sweep_number, A_soma=A_soma
        )

        obs["data"] = obs["data"] + junction_potential
        all_obs.append(obs)
        I = obs["I"]
        dt = obs["dt"]
        t_on = obs["t_on"]
        t_off = obs["t_off"]

        obs_stats = allen_obs_stats(
            data=obs,
            ephys_cell=ephys_cell,
            sweep_number=sweep_number,
            n_xcorr=n_xcorr,
            n_mom=n_mom,
            n_summary=n_summary,
        )
        obs_stats_ls.append(obs_stats)

    return obs_stats_ls, all_obs


def obs_params(reduced_model=False):
    """Parameters for x_o

    Parameters
    ----------
    reduced_model : bool
        If True, outputs two parameters
    Returns
    -------
    true_params : array
    labels_params : list of str
    """

    if reduced_model:
        true_params = np.array([50.0, 5.0])
    else:
        true_params = np.array([50.0, 5.0, 0.1, 0.07, 6e2, 60.0, 0.1, 70.0])

    labels_params = [
        "g_Na",
        "g_K",
        "g_leak",
        "g_M",
        "t_max",
        "-V_T",
        "noise",
        "-E_leak",
    ]
    labels_params = labels_params[0 : len(true_params)]

    return true_params, labels_params


def syn_current(
    duration=120, dt=0.01, t_on=10, step_current=True, curr_level=5e-4, seed=None
):
    t_offset = 0.0
    duration = duration
    t_off = duration - t_on
    t = np.arange(0, duration + dt, dt)

    # external current
    A_soma = np.pi * ((70.0 * 1e-4) ** 2)  # cm2
    I = np.zeros_like(t)
    I[int(np.round(t_on / dt)) : int(np.round(t_off / dt))] = (
        curr_level / A_soma
    )  # muA/cm2
    if step_current is False:
        rng_input = np.random.RandomState(seed=seed)

        times = np.linspace(0.0, duration, int(duration / dt) + 1)
        I_new = I * 1.0
        tau_n = 3.0
        nois_mn = 0.2 * I
        nois_fact = 2 * I * np.sqrt(tau_n)
        for i in range(1, times.shape[0]):
            I_new[i] = (
                I_new[i - 1]
                + dt
                * (
                    -I_new[i - 1]
                    + nois_mn[i - 1]
                    + nois_fact[i - 1] * rng_input.normal(0) / (dt**0.5)
                )
                / tau_n
            )
        I = I_new * 1.0

    return I, t_on, t_off, dt


def syn_obs_data(I, dt, params, V0=-70, seed=None, cython=False):
    """Data for x_o"""
    m = hh.HodgkinHuxley(I=I, dt=dt, V0=V0, cython=cython)
    return m.gen_single(params, seed=seed)


def syn_obs_stats(
    I,
    params,
    dt,
    t_on,
    t_off,
    data=None,
    V0=-70,
    n_xcorr=5,
    n_mom=5,
    n_summary=10,
    seed=None,
    cython=False,
):
    """Summary stats for x_o"""

    if data is None:
        m = hh.HodgkinHuxley(I=I, dt=dt, V0=V0, cython=cython)
        data = m.gen_single(params, seed=seed)

    s = HodgkinHuxleyStatsMoments(
        t_on, t_off, n_xcorr=n_xcorr, n_mom=n_mom, n_summary=n_summary
    )
    return s.calc([data])


def allen_obs_data(
    ephys_cell=464212183, sweep_number=33, A_soma=np.pi * (70.0 * 1e-4) ** 2
):
    """Data for x_o. Cell from AllenDB
    Parameters
    ----------
    ephys_cell : int
        Cell identity from AllenDB
    sweep_number : int
        Stimulus identity for cell ephys_cell from AllenDB
    """
    t_offset = 815.0
    duration = 1450.0
    dir_cache = os.path.dirname(inspect.getfile(hh.HodgkinHuxley))
    real_data_path = (
        dir_cache
        + "/support_files/ephys_cell_{}_sweep_number_{}.pkl".format(
            ephys_cell, sweep_number
        )
    )
    if not os.path.isfile(real_data_path):
        from allensdk.core.cell_types_cache import CellTypesCache
        from allensdk.api.queries.cell_types_api import CellTypesApi

        manifest_file = "cell_types/manifest.json"

        cta = CellTypesApi()
        ctc = CellTypesCache(manifest_file=manifest_file)
        data_set = ctc.get_ephys_data(ephys_cell)
        sweep_data = data_set.get_sweep(
            sweep_number
        )  # works with python2 and fails with python3
        sweeps = cta.get_ephys_sweeps(ephys_cell)

        sweep = sweeps[sweep_number]

        index_range = sweep_data["index_range"]
        i = sweep_data["stimulus"][0 : index_range[1] + 1]  # in A
        v = sweep_data["response"][0 : index_range[1] + 1]  # in V
        sampling_rate = sweep_data["sampling_rate"]  # in Hz
        dt = 1e3 / sampling_rate  # in ms
        i *= 1e6  # to muA
        v *= 1e3  # to mV
        v = v[int(t_offset / dt) : int((t_offset + duration) / dt)]
        i = i[int(t_offset / dt) : int((t_offset + duration) / dt)]

        real_data_obs = np.array(v).reshape(1, -1, 1)
        I_real_data = np.array(i).reshape(-1)
        t_on = int(sweep["stimulus_start_time"] * sampling_rate) * dt - t_offset
        t_off = (
            int(
                (sweep["stimulus_start_time"] + sweep["stimulus_duration"])
                * sampling_rate
            )
            * dt
            - t_offset
        )

        with open(real_data_path, "wb") as handle:
            pickle.dump((real_data_obs, I_real_data, dt, t_on, t_off), handle)
    else:

        def pickle_load(file):
            """Loads data from file."""
            f = open(file, "rb")
            data = pickle.load(f, encoding="latin1")
            f.close()
            return data

        real_data_obs, I_real_data, dt, t_on, t_off = pickle_load(real_data_path)

    t = np.arange(0, duration, dt)

    # external current
    I = I_real_data / A_soma  # muA/cm2

    # return real_data_obs, I_obs
    return {
        "data": real_data_obs.reshape(-1),
        "time": t,
        "dt": dt,
        "I": I.reshape(-1),
        "t_on": t_on,
        "t_off": t_off,
    }


def allen_obs_stats(
    data=None,
    ephys_cell=464212183,
    sweep_number=33,
    n_xcorr=5,
    n_mom=5,
    n_summary=13,
):
    """Summary stats for x_o. Cell from AllenDB
    Parameters
    ----------
    ephys_cell : int
        Cell identity from AllenDB
    sweep_number : int
        Stimulus identity for cell ephys_cell from AllenDB
    """

    if data is None:
        data = allen_obs_data(ephys_cell=ephys_cell, sweep_number=sweep_number)

    t_on = data["t_on"]
    t_off = data["t_off"]

    s = HodgkinHuxleyStatsMoments(
        t_on, t_off, n_xcorr=n_xcorr, n_mom=n_mom, n_summary=n_summary
    )
    return s.calc([data])


def resting_potential(data, dt, t_on):
    """Resting potential estimated from x_o"""
    return np.mean(data[0 : int(t_on / dt) - 5])


def prior(
    true_params, prior_uniform=True, prior_extent=False, prior_log=False, seed=None
):
    """Prior"""
    if not prior_extent:
        range_lower = param_transform(prior_log, 0.5 * true_params)
        range_upper = param_transform(prior_log, 1.5 * true_params)
    else:
        range_lower = param_transform(
            prior_log, np.array([0.5, 1e-4, 1e-4, 1e-4, 50.0, 40.0, 1e-4, 35.0])
        )
        range_upper = param_transform(
            prior_log, np.array([80.0, 15.0, 0.6, 0.6, 3000.0, 90.0, 0.15, 100.0])
        )

        range_lower = range_lower[0 : len(true_params)]
        range_upper = range_upper[0 : len(true_params)]

    if prior_uniform:
        prior_min = range_lower
        prior_max = range_upper
        return BoxUniform(prior_min, prior_max)
    else:
        prior_mn = param_transform(prior_log, true_params)
        prior_cov = np.diag((range_upper - range_lower) ** 2) / 12
        return MultivariateNormal(prior_mn, prior_cov)


def param_transform(prior_log, x):
    if prior_log:
        return np.log(x)
    else:
        return x


def param_invtransform(prior_log, x):
    if prior_log:
        return np.exp(x)
    else:
        return x
