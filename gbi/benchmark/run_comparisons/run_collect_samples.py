import torch
import numpy as np
from os import listdir, path, getcwd
import itertools

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import gbi.utils.utils as gbi_utils


# TO DO: keep betas for tasks in a config file somewhere
task_betas = {
    "uniform_1d": ["4", "20", "100"],
    "two_moons": ["10", "100", "1000"],
    "linear_gaussian": ["1", "10", "100"],
    "gaussian_mixture": ["10","25","75"],
}


@hydra.main(version_base="1.1", config_path="config", config_name="run_collect")
def collect_samples(cfg: DictConfig) -> None:
    # Define paths
    original_cwd = get_original_cwd()
    gt_dir = original_cwd + "/../../../results/benchmark/ground_truths/"
    inference_dir = original_cwd + "/../../../results/benchmark/algorithms/"
    xo_dir = original_cwd + "/../../../gbi/benchmark/tasks/"

    # load xo, gt posterior samples, and inference samples
    task_name = cfg.task.name
    betas = task_betas[task_name]
    xo_path = f"{xo_dir}/{task_name}/xos/"

    # Take the latest runs for GT and inference samples, unless specified.
    gt_datetime = cfg.gt_datetime
    if gt_datetime == "None":
        # Take last GT run
        gt_datetime = np.sort(listdir(f"{gt_dir}/{task_name}/"))[-1]
    print(f"Collecting GT posterior samples for {task_name} at {gt_datetime}.")

    # Note that this works best if all the inference objects are saved in the same directory, i.e., run at the same time, such that hydra would place them in the same datetime folder.
    # Otherwise specify different inference_datetime for each algo
    inference_datetime = cfg.inference_datetime
    print(
        f"Collecting learned posterior samples for {task_name} at {inference_datetime}."
    )
    algos = cfg.algos

    # Define all permutatations.
    xo_combs = [range(10), ["specified", "misspecified"], ["known", "unknown"]]
    xo_info_combs = list(itertools.product(*xo_combs))

    # Loop through to collect.
    posterior_samples_collected = []
    for ix_, xo_info in enumerate(xo_info_combs):
        # load xos and gt theta
        xos = gbi_utils.pickle_load(f"{xo_path}/xo_{xo_info[1]}_{xo_info[2]}.pkl")
        if xo_info[1] == "specified":
            theta_gts = gbi_utils.pickle_load(f"{xo_path}/theta_gt_{xo_info[2]}.pkl")
        else:
            theta_gts = torch.ones_like(xos) * torch.nan

        xo = xos[xo_info[0]]
        theta_gt = theta_gts[xo_info[0]]

        # load gt posterior samples
        posterior_samples = {}

        # Take the latest run
        posterior_samples["GT"] = {}
        for beta in betas:
            gt_sample_path = f"{gt_dir}/{task_name}/{gt_datetime}/beta_{beta}/obs_{xo_info[0]}_{xo_info[1]}_{xo_info[2]}/rejection_samples.pkl"
            if path.exists(gt_sample_path):
                posterior_samples["GT"][f"beta_{beta}"] = (
                    gbi_utils.pickle_load(gt_sample_path),
                    gt_datetime,
                )
            else:
                print(f"---GT posterior samples for beta={beta}, {xo_info} not found.")

        # load inference samples
        for algo in algos:
            posterior_dir = f"{inference_dir}/{task_name}/{inference_datetime}/{algo}/posterior_inference/"

            if path.isdir(posterior_dir):
                # Take the latest run
                posterior_datetime = np.sort(listdir(posterior_dir))[-1]
                # print(f"Collecting {algo} posterior samples for {xo_info} at {posterior_datetime}.")
                posterior_samples[algo] = {}

                for beta in betas:
                    if algo in ["NPE", "NLE"]:
                        # NPE and NLE will always have the same beta=1
                        beta = 1
                    elif algo in ["GBI", "eGBI", "ABC"]:
                        # GBI, eGBI, or ABC: collect all betas
                        pass

                    ps_path = f"{posterior_dir}/{posterior_datetime}/beta_{beta}/obs_{xo_info[0]}_{xo_info[1]}_{xo_info[2]}/posterior_samples.pkl"
                    if path.exists(ps_path):
                        posterior_samples[algo][f"beta_{beta}"] = (
                            gbi_utils.pickle_load(ps_path),
                            posterior_datetime,
                        )
                    else:
                        print(
                            f"---Posterior samples for {algo}, beta={beta}, {xo_info} not found."
                        )

            else:
                print(f"Posterior samples for {algo} not found.")

        posterior_samples_collected.append(
            [xo_info, {"xo": xo, "theta_gt": theta_gt}, posterior_samples]
        )

    # Save collected samples
    save_path = getcwd() + "/posterior_samples_collected.pkl"
    gbi_utils.pickle_dump(save_path, posterior_samples_collected)

    print("---Collected posterior samples:---")
    # [print(alg, list(v.keys())) for alg, v in posterior_samples_collected[0][2].items()]

    # Count the number of posterior samples collected for each algo and beta
    counters = {}
    for ix_, posterior_samples in enumerate(posterior_samples_collected):
        for algo, v in posterior_samples[2].items():
            if algo not in counters:
                counters[algo] = {}
            for beta, v2 in v.items():
                if beta not in counters[algo]:
                    counters[algo][beta] = 0
                counters[algo][beta] += 1
    [print(k, v) for k, v in counters.items()]
    print(f"All posterior samples saved as: {save_path}")
    return


if __name__ == "__main__":
    collect_samples()
