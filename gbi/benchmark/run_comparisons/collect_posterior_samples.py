import torch
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from os import listdir, path
from sbi.analysis import pairplot
from sbi.utils.metrics import c2st as C2ST
from scipy import stats
import itertools

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path

from gbi import distances
from gbi.benchmark.tasks.uniform_1d.task import UniformNoise1D
from gbi.benchmark.tasks.two_moons.task import TwoMoonsGBI
from gbi.benchmark.tasks.linear_gaussian.task import LinearGaussian
from gbi.benchmark.tasks.gaussian_mixture.task import GaussianMixture
import gbi.utils.utils as gbi_utils

gt_dir = "../results/benchmark/ground_truths/"
inference_dir = "../results/benchmark/algorithms/"
xo_dir = "../gbi/benchmark/tasks/"

task_classes = {
    "uniform_1d": UniformNoise1D,
    "two_moons": TwoMoonsGBI,
    "linear_gaussian": LinearGaussian,
    "gaussian_mixture": GaussianMixture
}
task_betas = {
    "uniform_1d": ["4","20","100"],
    "two_moons": ["10","100","1000"],
    "linear_gaussian": ["0.1","1.0","10.0"],
    "gaussian_mixture": ["2.0","10.0","50.0"],    
}
algos = ['GBI', 'NPE', 'NLE']


@hydra.main(version_base="1.1", config_path="config", config_name="collect_samples")
def collect_samples(cfg: DictConfig) -> None:
    # load xo, gt posterior samples, and inference samples
    # task_name = "uniform_1d"
    task_name = cfg.task_name
    
    betas = task_betas[task_name]
    xo_path = f"{xo_dir}/{task_name}/xos/"
    
    # Take the latest runs for GT and inference samples, unless specified.
    if cfg.gt_datetime==None:
        gt_datetime = np.sort(listdir(f"{gt_dir}/{task_name}/"))[-1]
    if cfg.inference_datetime==None:
        inference_datetime = np.sort(listdir(f"{inference_dir}/{task_name}/"))[-1]

    # Define all permutatations.
    xo_combs = [range(10), ['specified', 'misspecified'], ['known', 'unknown']]
    xo_info_combs = list(itertools.product(*xo_combs)) 

    # Loop through to collect.
    posterior_samples_collected = []
    for xo_info in xo_info_combs:
        # load xos and gt theta        
        xos = gbi_utils.pickle_load(f"{xo_path}/xo_{xo_info[1]}_{xo_info[2]}.pkl")
        if xo_info[1]=='specified':
            theta_gts = gbi_utils.pickle_load(f"{xo_path}/theta_gt_{xo_info[2]}.pkl")            
        else:
            theta_gts = torch.ones_like(xos)*torch.nan

        xo = xos[xo_info[0]]
        theta_gt = theta_gts[xo_info[0]]

        # load gt posterior samples
        posterior_samples = {}

        # Take the latest run
        posterior_samples['GT'] = {}
        for beta in betas:
            gt_path = f"{gt_dir}/{task_name}/{gt_datetime}/beta_{beta}/obs_{xo_info[0]}_{xo_info[1]}_{xo_info[2]}"
            posterior_samples['GT'][f"beta_{beta}"] = gbi_utils.pickle_load(gt_path + "/rejection_samples.pkl")
            
        # load inference samples
        for algo in algos:
            posterior_dir = f"{inference_dir}/{task_name}/{inference_datetime}/{algo}/posterior_inference/"    
            # Load inference algorithm
            if algo == "GBI":
                gbi_inference = gbi_utils.pickle_load(f"{inference_dir}/{task_name}/{inference_datetime}/{algo}/inference.pickle")
                
            if path.isdir(posterior_dir):
                # Take the latest run
                posterior_datetime = np.sort(listdir(posterior_dir))[-1]
                posterior_samples[algo] = {}

                for beta in betas:
                    ps_path = f"{posterior_dir}/{posterior_datetime}/beta_{beta}/obs_{xo_info[0]}_{xo_info[1]}_{xo_info[2]}/posterior_samples.pkl"                    
                    if path.exists(ps_path):
                        # posterior sample exists, load                        
                        if algo == "GBI":
                            posterior_samples[algo][f"beta_{beta}"] = gbi_utils.pickle_load(ps_path)
                        else:
                            posterior_samples[algo] = gbi_utils.pickle_load(ps_path)

        posterior_samples_collected.append([xo_info, {"xo": xo, "theta_gt": theta_gt,}, posterior_samples])
        print(posterior_samples_collected)
    
    return

if __name__ == "__main__":
    collect_samples()
