import pickle
from torch import zeros, ones, float32, tensor, as_tensor, eye

from gbi.tasks.linear_gaussian.ground_truth import Task
from sbi.inference.potentials.base_potential import BasePotential
from sbi.inference import MCMCPosterior
from sbi.utils import mcmc_transform


class PotentialFN(BasePotential):
    allow_iid_x = False  # type: ignore

    def __init__(self, prior, x_o, device, task):
        super().__init__(prior=prior, x_o=x_o, device=device)
        self.task = task

    def __call__(self, theta, track_gradients=False):
        return self.task.potential(theta)


dim = 10
x_o = zeros((dim,))

task = Task(dim=dim, x_o=x_o, seed=0)
potential_fn = PotentialFN(task.prior, x_o=x_o, device="cpu", task=task)
transform = mcmc_transform(task.prior, device="cpu")

posterior = MCMCPosterior(
    potential_fn=potential_fn,
    proposal=task.prior,
    theta_transform=transform,
    method="slice_np_vectorized",
    thin=10,
    warmup_steps=50,
    num_chains=100,
    init_strategy="resample",
)
samples = posterior.sample((100_000,))

with open(
    "../../results/ground_truths/linear_gaussian/mcmc_samples.pkl", "wb"
) as handle:
    pickle.dump(samples, handle)
