from typing import Tuple

import torch
from torch import tensor, as_tensor, float32, ones, zeros, eye, randn, mm, Tensor
from torch.distributions import MultivariateNormal

dim = 10
likelihood_shift = randn((dim,))
likelihood_cov = randn((dim,)) * eye(dim)
x_o = randn((dim,))
prior_mean = randn((dim,))
prior_cov = randn((dim,)) * eye(dim)

prior = MultivariateNormal(prior_mean, prior_cov)

beta = 1.0

def linear_gaussian(theta: Tensor) -> Tensor:
    """Simulator.
    """
    chol_factor = torch.linalg.cholesky(likelihood_cov)
    return likelihood_shift + theta + torch.mm(chol_factor, torch.randn_like(theta).T).T


def distance_fn(theta):
    """Computes E_{x|t}[(x - x_o)^2].
    """
    predicted_mean =likelihood_shift + theta
    expected_value_of_d = likelihood_cov.diagonal() + predicted_mean**2 - 2 * x_o * predicted_mean + x_o**2
    return expected_value_of_d


def potential(theta):
    """Potential for GBI ground truth posterior.
    """
    term1 = -beta * distance_fn(theta)
    return term1 + prior.log_prob(theta)


def true_posterior_linear_gaussian(x_o: Tensor) -> MultivariateNormal:
    """Returns the Bayes posterior.
    """

    num_trials, *_ = x_o.shape
    x_o_mean = x_o.mean(0)
    likelihood_mean = x_o_mean - likelihood_shift

    product_mean, product_cov = multiply_gaussian_pdfs(
        likelihood_mean, 1 / num_trials * likelihood_cov, prior_mean, prior_cov
    )

    posterior_dist = MultivariateNormal(product_mean, product_cov)

    return posterior_dist


def multiply_gaussian_pdfs(mu1, s1, mu2, s2) -> Tuple[Tensor, Tensor]:
    """Util for computing the Bayes posterior.
    """
    inv_s1s2 = torch.inverse(s1 + s2)

    # posterior mean = s2 * inv_s1pluss2 * mu1 + s1 * inv_s1pluss2 * mu2
    product_mean = torch.mv(torch.mm(s2, inv_s1s2), mu1) + torch.mv(
        torch.mm(s1, inv_s1s2), mu2
    )

    # posterior cov = s1 * inv_s1pluss2 * s2
    product_cov = torch.mm(torch.mm(s1, inv_s1s2), s2)

    return product_mean, product_cov
