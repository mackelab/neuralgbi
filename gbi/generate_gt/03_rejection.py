import pickle
from torch import zeros, Size

from gbi.tasks.linear_gaussian.ground_truth import Task
from sbi.inference.potentials.base_potential import BasePotential
from sbi.inference import RejectionPosterior


class PotentialFN(BasePotential):
    allow_iid_x = False  # type: ignore

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


dim = 10
x_o = zeros((dim,))
task = Task(dim=dim, x_o=x_o, seed=0)

with open("../../results/ground_truths/linear_gaussian/trained_nn.pkl", "rb") as handle:
    trained_nn = pickle.load(handle)

proposal = ProposalClass(trained_nn)
potential_fn = PotentialFN(task.prior, x_o=x_o, device="cpu", task=task)

posterior = RejectionPosterior(
    potential_fn=potential_fn,
    proposal=proposal,
)
samples = posterior.sample((10_000,))

with open(
    "../../results/ground_truths/linear_gaussian/rejection_samples.pkl", "wb"
) as handle:
    pickle.dump(samples, handle)
