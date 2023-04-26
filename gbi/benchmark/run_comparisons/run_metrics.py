import torch
import numpy as np
import pandas as pd
import itertools

import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from os import listdir, getcwd, path

from sbi.utils.metrics import c2st as C2ST
from gbi import distances
from gbi.benchmark.tasks.uniform_1d.task import UniformNoise1D
from gbi.benchmark.tasks.two_moons.task import TwoMoonsGBI
from gbi.benchmark.tasks.linear_gaussian.task import LinearGaussian
from gbi.benchmark.tasks.gaussian_mixture.task import GaussianMixture
import gbi.utils.utils as gbi_utils
import logging
import time

log = logging.getLogger("metrics")


task_classes = {
    "uniform_1d": UniformNoise1D,
    "two_moons": TwoMoonsGBI,
    "linear_gaussian": LinearGaussian,
    "gaussian_mixture": GaussianMixture,
}


@hydra.main(version_base="1.1", config_path="config", config_name="run_metrics")
def collect_metrics(cfg: DictConfig) -> None:
    # Load posterior samples and GBI object for computing distances
    posterior_samples_collected = gbi_utils.pickle_load(
        f"../posterior_samples_collected.pkl"
    )
    if cfg.algorithm == "COLLECT":
        df_summary, posterior_predictives_collected = [], []
        for ix_, psc in enumerate(posterior_samples_collected):
            xo_info, xo_theta, _ = psc
            cur_folder = f"../obs_{xo_info[0]}_{xo_info[1]}_{xo_info[2]}/"
            if path.exists(cur_folder):
                # Load summary and posterior predictives
                df_summary += [
                    pd.read_csv(cur_folder + f, index_col=0)
                    for f in listdir(cur_folder)
                    if "summary" in f
                ]
                posterior_predictives_collected.append(
                    [
                        xo_info,
                        xo_theta,
                        {
                            k: v
                            for f in listdir(cur_folder)
                            if "posterior_predictives" in f
                            for k, v in gbi_utils.pickle_load(cur_folder + f).items()
                        },
                    ]
                )
            else:
                log.info(f"Folder {cur_folder} does not exist")

        df_summary = pd.concat(df_summary, ignore_index=True)
        df_summary.to_csv(f"../summary_collected.csv")
        gbi_utils.pickle_dump(
            f"../posterior_predictives_collected.pkl", posterior_predictives_collected
        )
        return

    task_name = cfg.task.name
    if task_name == "gaussian_mixture":
        distance_func = distances.mmd_dist
    else:
        distance_func = distances.mse_dist

    print(
        f"Computing metrics on inference results from {cfg.inference_datetime}, collected on {cfg.collection_datetime}"
    )

    gbi_inference = gbi_utils.pickle_load(f"../../..//GBI/inference.pickle")

    # Find the corresponding index in the posterior samples collected
    xo_info = (cfg.task.xo_index, cfg.task.is_specified, cfg.task.is_known)
    idx_group = [psc[0] for psc in posterior_samples_collected].index(xo_info)

    xo_info, xo_theta, posterior_samples = posterior_samples_collected[idx_group]
    xo, theta_gt = xo_theta["xo"], xo_theta["theta_gt"]
    predictives_cur_xo = {}
    df_collect = []
    time_start = time.time()

    alg = cfg.algorithm
    samples_alg = posterior_samples[alg]
    if samples_alg:
        predictives_cur_xo[alg] = {}
        for beta_str, samples in samples_alg.items():
            beta = float(beta_str.split("_")[-1])
            task = task_classes[task_name](beta=beta, x_o=xo)

            # Compute predictives and summary, collect
            predictives = gbi_utils.compute_predictives(
                samples[0], task, distance_func, gbi_inference
            )
            predictives_cur_xo[alg][beta_str] = predictives

            summary = gbi_utils.compute_moments(predictives)
            df_summary = pd.DataFrame([summary.values()], columns=summary.keys())

            # Compute C2ST against GT GBI posterior if it's a GBI algorithm
            df_summary["c2st"] = torch.nan
            if (
                (alg in ["GBI", "eGBI"])
                and (beta_str in posterior_samples["GT"])
                and (cfg.compute_c2st == True)
            ):
                df_summary["c2st"] = C2ST(
                    samples[0], posterior_samples["GT"][beta_str][0]
                ).numpy()

            df_info = pd.DataFrame(
                [[task_name, *xo_info, alg, beta, samples[1]]],
                columns=[
                    "task",
                    "xo_idx",
                    "xo_specified",
                    "xo_known",
                    "algorithm",
                    "beta",
                    "samples_datetime",
                ],
            )
            df_collect.append(pd.concat((df_info, df_summary), axis=1))
            log.info(
                f"Finished {alg} {beta_str} for {xo_info} in {time.time() - time_start:.2f} seconds"
            )
            time_start = time.time()

        df_summaries = pd.concat(df_collect, ignore_index=True)
        df_summaries.to_csv(f"./summary_{alg}.csv")
        gbi_utils.pickle_dump(f"./posterior_predictives_{alg}.pkl", predictives_cur_xo)
    else:
        log.info(f"No samples for {alg} for {xo_info}")
    return


if __name__ == "__main__":
    collect_metrics()
