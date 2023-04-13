import torch
import numpy as np
import pandas as pd

import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from os import listdir

from sbi.utils.metrics import c2st as C2ST
from gbi import distances
from gbi.benchmark.tasks.uniform_1d.task import UniformNoise1D
from gbi.benchmark.tasks.two_moons.task import TwoMoonsGBI
from gbi.benchmark.tasks.linear_gaussian.task import LinearGaussian
from gbi.benchmark.tasks.gaussian_mixture.task import GaussianMixture
import gbi.utils.utils as gbi_utils

task_classes = {
    "uniform_1d": UniformNoise1D,
    "two_moons": TwoMoonsGBI,
    "linear_gaussian": LinearGaussian,
    "gaussian_mixture": GaussianMixture
}

@hydra.main(version_base="1.1", config_path="config", config_name="run_comparisons")
def collect_metrics(cfg: DictConfig) -> None:
    inference_dir = "../../../results/benchmark/algorithms/"
    task_name = cfg.task.name
    if task_name == "gaussian_mixture":
        distance_func = distances.mmd_dist
    else:
        distance_func = distances.mse_dist

    inference_datetime = cfg.inference_datetime
    if inference_datetime=="None":
        inference_datetime = np.sort(listdir(f"{inference_dir}/{task_name}/"))[-1]

    # Load posterior samples and GBI object for computing distances
    posterior_samples_collected = gbi_utils.pickle_load(f'{inference_dir}/{task_name}/{inference_datetime}/posterior_samples_all.pkl')
    gbi_inference = gbi_utils.pickle_load(f"{inference_dir}/{task_name}/{inference_datetime}/GBI/inference.pickle")

    df_collect = []
    for i_x in tqdm(range(len(posterior_samples_collected))):
        xo_info, xo_theta, posterior_samples = posterior_samples_collected[i_x]
        xo, theta_gt = xo_theta['xo'], xo_theta['theta_gt']

        for alg, samples_alg in posterior_samples.items():
            for beta_str, samples in samples_alg.items():
                beta = float(beta_str.split('_')[-1])
                task = task_classes[task_name](beta=beta, x_o=xo)            

                # compute predictives and summary                
                predictives = gbi_utils.compute_predictives(samples, task, distance_func, gbi_inference)
                summary = gbi_utils.compute_moments(predictives)
                # df_summary = pd.DataFrame(summary, dtype=float)
                df_summary = pd.DataFrame([summary.values()], columns=summary.keys())
                # compute C2ST against GT GBI posterior if it's a GBI algorithm
                df_summary['c2st'] = torch.nan  
                if alg in ["GBI", "eGBI", "ABC"]:                                        
                    df_summary['c2st'] = C2ST(samples, posterior_samples['GT'][beta_str])

                df_info = pd.DataFrame([[task_name, *xo_info, alg, beta]], columns=['task', 'xo_idx', 'xo_specified', 'xo_known', 'algorithm', 'beta'])
                df_collect.append(pd.concat((df_info, df_summary), axis=1))    

    df_summaries = pd.concat(df_collect, ignore_index=True)
    save_path = f'{inference_dir}/{task_name}/{inference_datetime}/comparison_summaries.csv'
    print(f"Summary CSV saved as {save_path}")
    df_summaries.to_csv(save_path)
    return

if __name__ == "__main__":
    collect_metrics()
