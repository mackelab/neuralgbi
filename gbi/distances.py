### Module for distance functions (sample-based), to be used during training
from torch import Tensor
import torch

from gbi.utils.mmd import sample_based_mmd


## MSE
def mse_dist(xs: Tensor, x_o: Tensor) -> Tensor:
    # Shape of xs should be [num_thetas, num_xs, num_x_dims].
    mse = ((xs - x_o) ** 2).mean(dim=2)  # Average over data dimensions.
    return mse.mean(dim=1)  # Monte-Carlo average


## MAE
def mae_dist(xs: Tensor, x_o: Tensor) -> Tensor:
    # Shape of xs should be [num_thetas, num_xs, num_x_dims].
    mae = ((xs - x_o).abs()).mean(dim=2)  # Average over data dimensions.
    return mae.mean(dim=1)  # Monte-Carlo average


## MMD
def mmd_dist(xs: Tensor, x_o: Tensor) -> Tensor:
    # TODO: need to clean up the shape checks here.

    # Check that x_o is at least 2D, i.e., [num_xs, num_x_dims].
    assert len(x_o.shape) > 1

    # Check that xs is of dim [num_thetas, 1, num_xs, num_x_dims]
    assert len(xs.shape) == 4

    # Check that there are more than 1 data points, i.e., samples in x.
    assert xs.shape[2] > 1

    if xs.shape == x_o.shape:
        # If xs and x_o have identical shapes, compute pairwise MMDs.
        mmds = torch.stack(
            [
                sample_based_mmd(xs[i_x].squeeze(), x_o[i_x].squeeze())
                for i_x in range(xs.shape[0])
            ]
        )

    else:
        # If they don't have identical shapes, compute all pairwise between xs and xo
        if len(x_o.shape) > 2:
            x_o = x_o.squeeze()
        assert x_o.shape[0] > 1
        mmds = torch.stack([sample_based_mmd(x[0], x_o) for x in xs])

    return mmds


## OTHER STUFF
