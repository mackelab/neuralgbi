### Module for distance functions (sample-based), to be used during training
from torch import Tensor

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
    return 


## OTHER STUFF