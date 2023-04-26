from typing import List, Tuple, Optional

import torch
from torch import tensor, as_tensor, float32, ones, zeros, eye, randn, Tensor, exp, log
from torch.distributions import Distribution


def ground_truth_mmd(
    x: Tensor,
    dists_y: List[Distribution],
    y_limits: Tensor,
    y_res: int = 100,
    scale: float = 0.01,
):
    term1 = sample_based_mmd_marginal(x, x, scale=scale)
    term2 = sample_integral_mixed_mmd_marginal(
        x=x, y_dist=dists_y, y_limits=y_limits, y_res=y_res, scale=scale
    )
    term3 = integral_based_mmd_marginal(
        x_dist=dists_y,
        y_dist=dists_y,
        x_limits=y_limits,
        y_limits=y_limits,
        x_res=y_res,
        y_res=y_res,
        scale=scale,
    )
    return term1 + term3 - 2 * term2


def sample_based_mmd(x, y, scale: float = 0.01):
    term1 = sample_based_mmd_marginal(x, x, scale=scale)
    term2 = sample_based_mmd_marginal(x, y, scale=scale)
    term3 = sample_based_mmd_marginal(y, y, scale=scale)
    return term1 + term3 - 2 * term2


def sample_based_mmd_marginal(x, y, scale: float = 0.01):
    """Assumes diagonal likelihood and sums over each dimension. Sum turns into
    product because exp(sum) = prod(exp)
    """
    dim = x.shape[1]
    term = tensor(
        [
            sample_based_mmd_term(x[:, d : d + 1], y[:, d : d + 1], scale=scale)
            for d in range(dim)
        ]
    ).prod()
    return term


def sample_based_mmd_term(x, y, scale: float = 0.01):
    num_x = x.shape[0]
    num_y = y.shape[0]
    xo1 = x.repeat((num_y, 1))
    xo2 = y.repeat_interleave((num_x), dim=0)
    distances = exp(-scale * ((xo1 - xo2) ** 2).sum(dim=1))
    average_dist = distances.mean(dim=0)
    return average_dist


def integral_based_mmd_marginal(
    x_dist: List[Distribution],
    y_dist: List[Distribution],
    x_limits: Tensor,
    y_limits: Tensor,
    x_res: int = 100,
    y_res: int = 100,
    scale: float = 0.01,
):
    """Assumes diagonal likelihood and sums over each dimension. Sum turns into
    product because exp(sum) = prod(exp)
    """
    dim = len(x_dist)
    term = tensor(
        [
            integral_mmd_term(
                x_dist[d],
                y_dist[d],
                x_limits[d],
                y_limits[d],
                x_res=x_res,
                y_res=y_res,
                scale=scale,
            )
            for d in range(dim)
        ]
    ).prod()
    return term


def integral_mmd_term(
    x_dist: Distribution,
    y_dist: Distribution,
    x_limits: Tensor,
    y_limits: Tensor,
    x_res: int = 100,
    y_res: int = 100,
    scale: float = 0.01,
):
    x_range = torch.linspace(x_limits[0].item(), x_limits[1].item(), x_res).unsqueeze(1)
    y_range = torch.linspace(y_limits[0].item(), y_limits[1].item(), y_res).unsqueeze(1)
    x_repeat = x_range.repeat((y_res, 1))
    y_repeat = y_range.repeat_interleave((x_res), dim=0)
    probs_x = x_dist.log_prob(x_repeat).exp()
    probs_y = y_dist.log_prob(y_repeat).exp()
    distances = exp(-scale * ((x_repeat - y_repeat) ** 2).sum(dim=1))
    dx = (x_limits[1].item() - x_limits[0].item()) / (x_res - 1)
    dy = (y_limits[1].item() - y_limits[0].item()) / (y_res - 1)
    integral = (probs_x * probs_y * distances).sum() * dx * dy
    return integral


def sample_integral_mixed_mmd_marginal(
    x,
    y_dist: List[Distribution],
    y_limits: Tensor,
    y_res: int = 100,
    scale: float = 0.01,
):
    """Assumes diagonal likelihood and sums over each dimension. Sum turns into
    product because exp(sum) = prod(exp)
    """
    dim = len(y_dist)
    term = tensor(
        [
            sample_integral_mixed_mmd_term(
                x[:, d : d + 1], y_dist[d], y_limits[d], y_res=y_res, scale=scale
            )
            for d in range(dim)
        ]
    ).prod()
    return term


def sample_integral_mixed_mmd_term(
    x, y_dist: Distribution, y_limits: Tensor, y_res: int = 100, scale: float = 0.01
):
    y_range = torch.linspace(y_limits[0].item(), y_limits[1].item(), y_res).unsqueeze(1)
    y_repeat = y_range.repeat((x.shape[0], 1))
    probs_y = y_dist.log_prob(y_repeat).exp()
    probs_y = torch.reshape(probs_y, (y_res, x.shape[0]))
    y_reshape = torch.reshape(y_repeat, (y_res, x.shape[0], 1))
    distances = exp(-scale * ((x - y_reshape) ** 2).sum(dim=2))
    dy = (y_limits[1].item() - y_limits[0].item()) / (y_res - 1)
    integrals = (distances * probs_y).sum(dim=0) * dy
    monte_carlo_integral = torch.mean(integrals)
    return monte_carlo_integral
