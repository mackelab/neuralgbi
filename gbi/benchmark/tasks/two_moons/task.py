from typing import Tuple, Optional

import torch
from sbibm.tasks.two_moons.task import TwoMoons as TwoMoonsSBIBM
from torch import Tensor, linspace, meshgrid, diff, stack
from sbi.utils.torchutils import atleast_2d


class TwoMoonsGBI(TwoMoonsSBIBM):
    def __init__(
        self,
        beta: float = 1.0,
        seed: int = 0,
        x_o: Optional[Tensor] = None,
        x_limits: Tensor = Tensor([[-1.2, 0.4], [-1.6, 1.6]]),
        x_resolutions: Tensor = Tensor([500, 500]),
    ):
        """Recommended beta: [10, 100, 1000]"""
        # Subclass of Two Moons task from SBI benchmark.
        super().__init__()

        # Set seed.
        _ = torch.manual_seed(seed)

        # Inherit prior and simulator.
        self.prior = self.get_prior_dist()
        self.simulator = self.get_simulator()

        # Set x_o and temperature.
        self.x_o = x_o
        self.beta = beta

        # Set x-grid parameters
        # Prior predictibe bounds from 5 million simulations:
        # x1_lim = [-1.1590, 0.3892]
        # x2_lim = [-1.5145, 1.5274]
        self.x_limits = x_limits

        # 500x500 grid seems sufficient.
        self.x_resolutions = x_resolutions
        self.dx2 = None

    def simulate(self, theta: Tensor) -> Tensor:
        """Pass-through to use sbibm task simulator."""
        return self.simulator(theta)

    def set_x_o(self, x_o: Tensor):
        self.x_o = x_o
        # Recompute grid based on new x_o.
        self.distance_grid = (self.x_grid - self.x_o).pow(2).mean(axis=1)

    def distance_fn(self, theta: Tensor) -> Tensor:
        """Compute distance function, integrate over grid of x."""
        # Check x_o exists and that theta is 2D.
        assert self.x_o is not None
        theta = atleast_2d(theta)

        # Only make the x-grid and compute distance on the grid once.
        # NOTE: setting x_o manually does not trigger recomputing the grid
        # and therefore results in the WRONG grid! Use self.set_x_o().
        if self.dx2 == None:
            self.x_grid, self.dx2 = self.make_x_grid()
            self.distance_grid = (self.x_grid - self.x_o).pow(2).mean(axis=1)

        # Compute integral d(x,x_o)*p(x|theta)*dx^2 over the grid.
        # Adjust by factor of by because uniform likelihood model is unnormalized.
        integral = (
            stack(
                [
                    (
                        self.distance_grid
                        * self._likelihood(th, self.x_grid, log=True).exp()
                        * self.dx2
                    ).sum()
                    for th in theta
                ],
                dim=0,
            )
            * torch.pi
        )
        return integral

    def potential(self, theta: Tensor) -> Tensor:
        """Potential for GBI ground truth posterior."""
        term1 = -self.beta * self.distance_fn(theta)
        return term1 + self.prior.log_prob(theta)

    def make_x_grid(self) -> Tuple:
        """Helper function for making the x-grid."""
        x1_res = self.x_resolutions[0].to(int)
        x2_res = self.x_resolutions[1].to(int)
        x1 = linspace(self.x_limits[0][0], self.x_limits[0][1], x1_res)
        x2 = linspace(self.x_limits[1][0], self.x_limits[1][1], x2_res)
        xs = stack(meshgrid(x1, x2))
        x_grid = xs.reshape((2, (x1_res) * (x2_res))).T
        dx2 = diff(x1).mean() * diff(x2).mean()
        return x_grid, dx2


#     ### Code for doing distance computation in a more efficient way.
#     #     By translating x such that theta = [0,0], resulting in
#     #     needing only a single likelihood estimate.
#     def undo_translation(self, theta, x):
#         theta = atleast_2d(theta)
#         d1 = -(theta[:,0]+theta[:,1]).abs()/sqrt(2.)
#         d2 = (-theta[:,0]+theta[:,1])/sqrt(2.)
#         untranslated_x = x - torch.stack((d1,d2),dim=1)
#         return untranslated_x

#     def distance_fn_from_untranslated():
#         x_grid, dx2 = make_2d_grid((.2,.4), (-.2, .2), 1001,2001)
#         lh_grid = task._likelihood(Tensor([0,0]), x_grid, log=True).exp()
#         dist_shiftedfted = Tensor([((x_grid - task.undo_translation(th, task.x_o)).pow(2).mean(1) * lh_grid * dx2).sum() for th in theta])
