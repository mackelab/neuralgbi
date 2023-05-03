from typing import Any, Union
import pickle
from torch import zeros, Size

from sbi.inference.potentials.base_potential import BasePotential
from sbi.inference import RejectionPosterior
from sbi.utils import mcmc_transform


class PotentialFN(BasePotential):
    allow_iid_x = True  # type: ignore

    def __init__(self, prior, x_o, device, task):
        super().__init__(prior=prior, x_o=x_o, device=device)
        self.task = task

    def __call__(self, theta, track_gradients=False):
        return self.task.potential(theta)


class ProposalClass:
    def __init__(self, net):
        self.net = net

    def sample(self, sample_shape, **kwargs):
        samples = self.net.sample(Size(sample_shape).numel())
        return samples.detach()

    def log_prob(self, theta, **kwargs):
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        return self.net.log_prob(theta)

    def set_default_x(self, x_o):
        pass


def run_rejection(task, config, proposal: Union[str, Any]):
    if proposal == "net":
        with open("trained_nn.pkl", "rb") as handle:
            trained_nn = pickle.load(handle)
        proposal = ProposalClass(trained_nn)

    potential_fn = PotentialFN(task.prior, x_o=task.x_o, device="cpu", task=task)
    transform = mcmc_transform(task.prior, device="cpu")

    posterior = RejectionPosterior(
        potential_fn=potential_fn,
        theta_transform=transform,
        proposal=proposal,
        num_samples_to_find_max=config.num_samples_to_find_max,
        num_iter_to_find_max=config.num_iter_to_find_max,
    )
    samples = posterior.sample((config.num_rej_samples,))

    with open("rejection_samples.pkl", "wb") as handle:
        pickle.dump(samples, handle)
