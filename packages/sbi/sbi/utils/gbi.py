from typing import Callable

import torch
from torch import Tensor
from torch import nn
from sbi.inference.potentials.base_potential import BasePotential
from sbi.utils.torchutils import atleast_2d


def mse_dist(xs: Tensor, x_o: Tensor) -> Tensor:
    # Shape of xs should be [num_thetas, num_xs, num_x_dims]
    mse = ((xs - x_o) ** 2).mean(dim=2)  # Average data dimension.
    return mse.mean(dim=1)  # Monte-Carlo average


def build_generalized_log_likelihood(
    emulator: nn.Module, x_o: Tensor, dist_func: Callable, n_per_theta: int = 1_000
):
    def generalized_loglikelihood(thetas: Tensor):
        thetas = atleast_2d(thetas)
        emulated_x = emulator.sample(n_per_theta, context=thetas)
        distances = dist_func(emulated_x, x_o)
        assert distances.shape == (thetas.shape[0],)
        return distances

    return generalized_loglikelihood


class GBIPotential(BasePotential):
    allow_iid_x = False

    def __init__(self, prior, gen_llh_fn, beta=1.0):
        super().__init__(prior, None)
        self.gen_llh_fn = gen_llh_fn
        self.beta = beta

    def __call__(self, theta, track_gradients=True):
        with torch.set_grad_enabled(track_gradients):
            return -self.beta * self.gen_llh_fn(theta) + self.prior.log_prob(theta)
