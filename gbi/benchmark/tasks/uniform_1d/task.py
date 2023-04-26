from typing import Tuple, Optional, Union

import torch
from torch import tensor, as_tensor, float32, ones, zeros, eye, randn, Tensor
from sbi.utils import BoxUniform


class UniformNoise1D:
    def __init__(
        self,
        prior_bounds: Tuple = (-1.5, 1.5),
        beta: float = 1.0,
        seed: int = 0,
        poly_coeffs: Tensor = Tensor([0.1627, 0.9073, -1.2197, -1.4639, 1.4381]),
        epsilon: Union[Tensor, float] = 0.25,
        x_o: Optional[Tensor] = None,
    ):
        """Suggested beta: [4, 20, 100]"""
        # Set seed.
        _ = torch.manual_seed(seed)

        # Make prior for theta.
        self.prior = BoxUniform(prior_bounds[0] * ones(1), prior_bounds[1] * ones(1))

        # noise_likelihood model.
        self.noise_likelihood = BoxUniform(-epsilon * ones(1), epsilon * ones(1))

        # Set polynomial coefficients; default makes good curve.
        self.poly_coeffs = poly_coeffs

        # Set uniform noise half-width.
        self.epsilon = epsilon

        # Set GBI likelihood temperature.
        self.beta = beta

        # Set observation
        self.x_o = x_o

    def simulate_noiseless(self, theta: Tensor) -> Tensor:
        """Noiseless simulator."""
        return (
            torch.hstack([(0.8 * (theta + 0.25)) ** i for i in range(5)])
            * self.poly_coeffs
        ).sum(1, keepdim=True)

    def simulate(self, theta: Tensor) -> Tensor:
        """Simulator with U[-eps, eps] noise applied."""
        # Get uniform noise of [-epsilon, epsilon].
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        noise = self.noise_likelihood.sample((theta.shape[0],))  # type: ignore
        return self.simulate_noiseless(theta) + noise

    def distance_fn(self, theta):
        """Computes E_{x|t}[(x - x_o)^2]."""
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        x_true = self.simulate_noiseless(theta)
        x_low = x_true - self.epsilon
        x_high = x_true + self.epsilon
        expected_value_of_d = (
            self._eval_distance_integral(x_high) - self._eval_distance_integral(x_low)
        ).squeeze(1) / (2 * self.epsilon)
        return expected_value_of_d

    def potential(self, theta):
        """Potential for GBI ground truth posterior."""
        term1 = -self.beta * self.distance_fn(theta)
        return term1 + self.prior.log_prob(theta)

    def _eval_distance_integral(self, x):
        """Integral of (x-x_o)^2 (without constant)."""
        assert self.x_o is not None
        return (x ** 3) / 3 - self.x_o * x ** 2 + x * self.x_o ** 2

    # def true_posterior_linear_gaussian(self, x_o: Tensor) -> MultivariateNormal:
    #     """Returns the Bayes posterior."""

    #     num_trials, *_ = x_o.shape
    #     x_o_mean = x_o.mean(0)
    #     likelihood_mean = x_o_mean - self.likelihood_shift

    #     product_mean, product_cov = self.multiply_gaussian_pdfs(
    #         likelihood_mean,
    #         1 / num_trials * self.likelihood_cov,
    #         self.prior_mean,
    #         self.prior_cov,
    #     )

    #     posterior_dist = MultivariateNormal(product_mean, product_cov)

    #     return posterior_dist

    # def multiply_gaussian_pdfs(self, mu1, s1, mu2, s2) -> Tuple[Tensor, Tensor]:
    #     """Util for computing the Bayes posterior."""
    #     inv_s1s2 = torch.inverse(s1 + s2)

    #     # posterior mean = s2 * inv_s1pluss2 * mu1 + s1 * inv_s1pluss2 * mu2
    #     product_mean = torch.mv(torch.mm(s2, inv_s1s2), mu1) + torch.mv(
    #         torch.mm(s1, inv_s1s2), mu2
    #     )

    #     # posterior cov = s1 * inv_s1pluss2 * s2
    #     product_cov = torch.mm(torch.mm(s1, inv_s1s2), s2)

    #     return product_mean, product_cov
