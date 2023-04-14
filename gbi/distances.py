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
    assert len(xs.shape) == 4
    assert len(x_o.shape) > 1
    assert xs.shape[2] > 1
    assert x_o.shape[0] > 1

    mmds = torch.stack([sample_based_mmd(x[0], x_o) for x in xs])
    return mmds


## OTHER STUFF
