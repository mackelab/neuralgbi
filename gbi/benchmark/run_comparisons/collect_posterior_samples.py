import torch
import numpy as np
from os import listdir, path
import itertools

import hydra
from omegaconf import DictConfig
import gbi.utils.utils as gbi_utils

gt_dir = "../../../results/benchmark/ground_truths/"
inference_dir = "../../../results/benchmark/algorithms/"
xo_dir = "../../../gbi/benchmark/tasks/"

task_betas = {
    "uniform_1d": ["4", "20", "100"],
    "two_moons": ["10", "100", "1000"],
    "linear_gaussian": ["0.1", "1.0", "10.0"],
    "gaussian_mixture": ["2.0", "10.0", "50.0"],
}


@hydra.main(version_base="1.1", config_path="config", config_name="run_comparisons")
def collect_samples(cfg: DictConfig) -> None:
    # load xo, gt posterior samples, and inference samples
    task_name = cfg.task.name
    betas = task_betas[task_name]
    xo_path = f"{xo_dir}/{task_name}/xos/"

    # Take the latest runs for GT and inference samples, unless specified.
    gt_datetime = cfg.gt_datetime
    if gt_datetime == "None":
        gt_datetime = np.sort(listdir(f"{gt_dir}/{task_name}/"))[-1]

    inference_datetime = cfg.inference_datetime
    if inference_datetime == "None":
        inference_datetime = np.sort(listdir(f"{inference_dir}/{task_name}/"))[-1]

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
            gt_path = f"{gt_dir}/{task_name}/{gt_datetime}/beta_{beta}/obs_{xo_info[0]}_{xo_info[1]}_{xo_info[2]}"
            if path.isdir(gt_path):
                posterior_samples["GT"][f"beta_{beta}"] = gbi_utils.pickle_load(
                    gt_path + "/rejection_samples.pkl"
                )
            else: 
                print(f"---GT posterior samples for beta={beta}, {xo_info} not found.")

            

        # load inference samples
        for algo in algos:
            posterior_dir = f"{inference_dir}/{task_name}/{inference_datetime}/{algo}/posterior_inference/"

            if path.isdir(posterior_dir):
                # Take the latest run
                posterior_datetime = np.sort(listdir(posterior_dir))[-1]
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
                        posterior_samples[algo][f"beta_{beta}"] = gbi_utils.pickle_load(
                            ps_path
                        )
                    else:
                        print(
                            f"---Posterior samples for {algo}, beta={beta}, {xo_info} not found."
                        )

            else:
                print(f"Posterior samples for {algo} not found.")

        posterior_samples_collected.append(
            [
                xo_info,
                {
                    "xo": xo,
                    "theta_gt": theta_gt,
                },
                posterior_samples,
            ]
        )

    # Save collected samples
    save_path = (
        f"{inference_dir}/{task_name}/{inference_datetime}/posterior_samples_all_{task_name}.pkl"
    )
    gbi_utils.pickle_dump(save_path, posterior_samples_collected)
    print("---Collected posterior samples:---")
    [print(alg, list(v.keys())) for alg, v in posterior_samples_collected[0][2].items()]
    print(f"All posterior samples saved as: {save_path}")
    return


if __name__ == "__main__":
    collect_samples()
