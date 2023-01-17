from typing import List, Tuple, Optional

import torch
from torch import tensor, as_tensor, float32, ones, zeros, eye, randn, Tensor, exp, log
from torch.distributions import MultivariateNormal, Distribution, Normal
from sbi.utils import BoxUniform
from gbi.utils.mmd import ground_truth_mmd


class GaussianMixture:
    def __init__(
        self,
        limits: Tensor = tensor([[-14, 14], [-14, 14]]),
        resolution: int = 250,
        x_o: Optional[Tensor] = None,
        num_trials: int = 5,
        beta: float = 1.0,
        dim: int = 2,
        mmd_length_scale: float = 0.1,
    ):
        self.limits = limits
        self.resolution = resolution
        self.prior = BoxUniform(-10 * ones(dim), 10 * ones(dim))
        self.x_o = x_o
        self.num_trials = num_trials
        self.beta = beta
        self.mmd_length_scale = mmd_length_scale

    def simulate(self, theta: Tensor) -> Tensor:
        """Simulator."""
        samples1 = torch.randn((self.num_trials, *theta.shape)) + theta
        samples2 = 0.1 * torch.randn((self.num_trials, *theta.shape)) + theta
        all_samples = torch.zeros(*samples1.shape)

        bern = torch.bernoulli(0.5 * ones((self.num_trials, theta.shape[0]))).bool()

        all_samples[bern] = samples1[bern]
        all_samples[~bern] = samples2[~bern]
        all_samples = torch.permute(all_samples, (1, 0, 2))
        return all_samples

    def build_marginal_dist(self, predicted_mean):
        class MixtureDist(Distribution):
            def __init__(self, predicted_mean):
                super().__init__()
                self.dist1 = MultivariateNormal(tensor([predicted_mean]), eye(1))
                self.dist2 = MultivariateNormal(tensor([predicted_mean]), 0.01 * eye(1))

            def log_prob(self, x):
                prob1 = self.dist1.log_prob(x).exp()
                prob2 = self.dist1.log_prob(x).exp()
                return (0.5 * prob1 + 0.5 * prob2).log()

        marginals = [MixtureDist(p) for p in predicted_mean[0]]
        return marginals

    def distance_fn(self, theta):
        """Computes E_{x|t}[(x - x_o)^2]."""
        assert self.x_o is not None, "x_o not set."
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)

        marginals = self.build_marginal_dist(theta)
        mmd_x = ground_truth_mmd(
            x=self.x_o,
            dists_y=marginals,
            y_limits=self.limits,
            y_res=self.resolution,
            scale=self.mmd_length_scale,
        )
        return mmd_x

    def potential(self, theta):
        """Potential for GBI ground truth posterior."""
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)

        potentials = []
        for t in theta:
            term1 = -self.beta * self.distance_fn(t)
            potentials.append(term1 + self.prior.log_prob(t))
        return torch.stack(potentials)
