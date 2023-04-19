from typing import Callable, Optional
from torch import Tensor
from torch import exp, rand


class ABC:
    def __init__(self):
        pass

    def append_simulations(self, theta, x):
        self.theta = theta
        self.x = x
        return self

    def set_default_x(self, x_o):
        self.x_o = x_o
        return self

    def set_dist_fn(self, dist_fn):
        self.dist_fn = dist_fn
        return self

    def sample(self, beta: float, x: Optional[Tensor] = None):
        """Returns ABC samples. These are exact but their number is fixed.

        Args:
            distance_func: Takes two arguments and returns their distance. First
                argument can be batched, second argument should be batched with
                batchsize 1.
            beta:
            x: Use `self.x_o` if `None` is passed.
        """
        if x is None:
            obs = self.x_o
        else:
            obs = x

        distances = self.dist_fn(self.x.unsqueeze(1), obs)
        acceptance_probs = exp(-beta * distances)
        rands = rand(acceptance_probs.shape)
        accepted_sample = (rands < acceptance_probs).bool()
        return self.theta[accepted_sample]
