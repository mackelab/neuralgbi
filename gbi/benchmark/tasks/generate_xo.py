import pickle
from typing import Optional
import os
import sys
import torch
from pathlib import Path


def generate_x_specified(task, n_observations: int = 10):
    """Simulate x from simulator."""
    theta = task.prior.sample((n_observations,))
    x = task.simulate(theta)
    return theta, x

def generate_x_misspecified(
    task, n_observations: int = 10, diffusion_scale=0.5, max_steps=10000
):
    """Generate misspecified x by diffusion from simulated x."""
    theta = task.prior.sample((100_000,))
    x = task.simulate(theta)
    counter = 0
    x_min, x_max = x.min(0)[0], x.max(0)[0]
    x_miss = x[torch.randint(high=x.shape[0], size=(n_observations,))]
    mask = torch.zeros((x_miss.shape[0],))
    while (not (mask > 0).all()) and counter < max_steps:
        x_miss += (
            ((torch.randn((n_observations, x.shape[1])) > 0).to(float) - 0.5)
            * 2
            * x.std(0)
            * (1.0 - mask.unsqueeze(1))
            * diffusion_scale
        )
        mask = ((x_miss < x_min).any(1) | (x_miss > x_max).any(1)).to(float)
        # x_miss = x[rand_idx] + ((torch.randn((10,x.shape[1]))>0).to(float)-0.5)*2*x.std(0)*2
        counter += 1

    if counter - 1 == max_steps:
        print("Not all points out of bounds.")
    else:
        print(f"{counter-1} steps of diffusion for misspecified data.")

    return x_miss


def generate_xo(task_name="uniform_1d", n_observations=10):
    """Generate observations, for training and testing."""
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/" + task_name + "/xos/"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(dir_path)

    if task_name == "uniform_1d":
        from gbi.benchmark.tasks.uniform_1d.task import UniformNoise1D

        task = UniformNoise1D(seed=42)
    elif task_name == "two_moons":
        from gbi.benchmark.tasks.two_moons.task import TwoMoonsGBI

        task = TwoMoonsGBI(seed=42)
    elif task_name == "linear_gaussian":
        from gbi.benchmark.tasks.linear_gaussian.task import LinearGaussian

        task = LinearGaussian(seed=42)
    elif task_name == "gaussian_mixture":
        from gbi.benchmark.tasks.gaussian_mixture.task import GaussianMixture

        task = GaussianMixture(seed=42)
    else:
        raise NameError("Task doesn't exist.")

    # Generate observed (known) xos.
    theta_gt_known, xo_specified_known = generate_x_specified(task, n_observations)
    # Generate unobserved xos for test time.
    theta_gt_unknown, xo_specified_unknown = generate_x_specified(task, n_observations)

    if task_name == "gaussian_mixture":
        # Generate observed (known) xos.
        theta = task.prior.sample((n_observations,))
        xo_misspecified_known = task.simulate_misspecified(theta)
        # Generate unobserved xos for test time.
        theta = task.prior.sample((n_observations,))
        xo_misspecified_unknown = task.simulate_misspecified(theta)
    else:
        xo_misspecified_known = generate_x_misspecified(task, n_observations)
        xo_misspecified_unknown = generate_x_misspecified(task, n_observations)

    # print(xo_specified_known[0])
    # print(xo_specified_unknown[0])
    # print(xo_misspecified_known[0])
    # print(xo_misspecified_unknown[0])

    # Save theta_gt and xos
    pickle_dump(f"{dir_path}/theta_gt_known.pkl", theta_gt_known)
    pickle_dump(f"{dir_path}/xo_specified_known.pkl", xo_specified_known)
    pickle_dump(f"{dir_path}/xo_misspecified_known.pkl", xo_misspecified_known)

    # Save theta_gt and xos
    pickle_dump(f"{dir_path}/theta_gt_unknown.pkl", theta_gt_unknown)
    pickle_dump(f"{dir_path}/xo_specified_unknown.pkl", xo_specified_unknown)
    pickle_dump(f"{dir_path}/xo_misspecified_unknown.pkl", xo_misspecified_unknown)
    return


def pickle_dump(full_path, data_dump):
    with open(full_path, "wb") as handle:
        pickle.dump(data_dump, handle)


if __name__ == "__main__":
    task_name = sys.argv[1]
    n_observations = int(sys.argv[2])
    generate_xo(task_name, n_observations)
