import pickle
from typing import Optional
import os
import sys
import torch


# Make this script generate x for all tasks.

# def generate_xo():
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     _ = torch.manual_seed(0)
#     task = UniformNoise1D()
#     theta_gt = task.prior.sample((10,))
#     simulated_x = task.simulate(theta_gt)
#     with open(f"{dir_path}/xo.pkl", "wb") as handle:
#         pickle.dump(simulated_x, handle)
#     with open(f"{dir_path}/gt.pkl", "wb") as handle:
#         pickle.dump(theta_gt, handle)


def generate_x_specified(task, n_observations: int = 10):
    """Simulate x from simulator."""
    theta = task.prior.sample((n_observations,))
    x = task.simulate(theta)
    return theta, x

def generate_x_misspecified(task, n_observations: int = 10, diffusion_scale = 0.5, max_steps = 10000):
    """Generate misspecified x by diffusion from simulated x."""
    theta = task.prior.sample((100_000,))
    x = task.simulate(theta)
    counter = 0
    x_min, x_max = x.min(0)[0], x.max(0)[0]    
    x_miss = x[torch.randint(high=x.shape[0], size=(10,))]
    mask = torch.zeros((x_miss.shape[0],))
    while (not (mask>0).all()) and counter < max_steps:        
        x_miss += ((torch.randn((10,x.shape[1]))>0).to(float)-0.5)*2*x.std(0) * (1.-mask.unsqueeze(1)) * diffusion_scale
        mask = ((x_miss < x_min).any(1) | (x_miss > x_max).any(1)).to(float)
        # x_miss = x[rand_idx] + ((torch.randn((10,x.shape[1]))>0).to(float)-0.5)*2*x.std(0)*2
        counter +=1

    if counter-1==max_steps:
        print('Not all points out of bounds.')
    else:
        print(f"{counter-1} steps of diffusion for misspecified data.")

    return x_miss

def generate_xo(task_name='uniform_1d', n_observations=10):
    """Generate observations, for training and testing."""
    dir_path = os.path.dirname(os.path.realpath(__file__)) + '/' + task_name
    print(dir_path)

    if task_name == 'uniform_1d':
        from gbi.benchmark.tasks.uniform_1d.task import UniformNoise1D
        task = UniformNoise1D(seed=42)
    elif task_name == 'two_moons':
        from gbi.benchmark.tasks.two_moons.task import TwoMoonsGBI
        task = TwoMoonsGBI(seed=42)
    elif task_name == 'linear_gaussian':
        from gbi.benchmark.tasks.linear_gaussian.task import LinearGaussian
        task = LinearGaussian(seed=42)
    else:
        raise NameError("Task doesn't exist.")

    # Generate observed (known) xos.
    theta_gt_known, xo_specified_known = generate_x_specified(task, n_observations)
    xo_misspecified_known = generate_x_misspecified(task, n_observations)
    # Save theta_gt and xos
    pickle_dump(f"{dir_path}/theta_gt_known.pkl", theta_gt_known)
    pickle_dump(f"{dir_path}/xo_specified_known.pkl", xo_specified_known)
    pickle_dump(f"{dir_path}/xo_misspecified_known.pkl", xo_misspecified_known)

    # Generate unobserved xos for test time.
    theta_gt_unknown, xo_specified_unknown = generate_x_specified(task, n_observations)
    xo_misspecified_unknown = generate_x_misspecified(task, n_observations)
    # Save theta_gt and xos
    pickle_dump(f"{dir_path}/theta_gt_unknown.pkl", theta_gt_unknown)
    pickle_dump(f"{dir_path}/xo_specified_unknown.pkl", xo_specified_unknown)
    pickle_dump(f"{dir_path}/xo_misspecified_unknown.pkl", xo_misspecified_unknown)
    return


def pickle_dump(full_path, data_dump):
    with open(full_path, "wb") as handle:
        pickle.dump(data_dump, handle)        


# def return_xo(index: Optional[int] = None):
#     dir_path = os.path.dirname(os.path.realpath(__file__))

#     with open(f"{dir_path}/xo.pkl", "rb") as handle:
#         simulated_x = pickle.load(handle)
#     if index is not None:
#         return simulated_x[index].unsqueeze(0)
#     else:
#         return simulated_x


# def return_gt(index: Optional[int] = None):
#     dir_path = os.path.dirname(os.path.realpath(__file__))

#     with open(f"{dir_path}/gt.pkl", "rb") as handle:
#         simulated_x = pickle.load(handle)
#     if index is not None:
#         return simulated_x[index].unsqueeze(0)
#     else:
#         return simulated_x


if __name__ == "__main__":
    task_name = sys.argv[1]
    n_observations = int(sys.argv[2])
    generate_xo(task_name, n_observations)
