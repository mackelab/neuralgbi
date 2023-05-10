import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, Optional, Union
from copy import deepcopy
import numpy as np
from math import ceil
import torch
from torch import Tensor, nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Distribution

from sbi.utils.torchutils import atleast_2d
from sbi.inference.potentials.base_potential import BasePotential
from sbi.neural_nets.embedding_nets import PermutationInvariantEmbedding, FCEmbedding
from time import time
from pyknos.nflows.nn import nets
from sbi.utils.sbiutils import standardizing_net, DeStandardize, MultiplyByMean


class GBInference:
    def __init__(
        self,
        prior: Distribution,
        distance_func: Callable,
        do_precompute_distances: bool = True,
        n_lowest: Optional[int] = None,
    ):
        self.prior = prior
        self.distance_func = distance_func
        self.do_precompute_distances = do_precompute_distances
        self.n_lowest = n_lowest
        self.zscore_distance_precomputed = []

    def append_simulations(self, theta: Tensor, x: Tensor, x_target: Tensor):
        """Append simulation data: theta, x, and target x."""
        self.theta = theta
        self.x = x
        self.x_target = x_target
        if self.do_precompute_distances:
            # Pre compute the distance function between all x and x_targets.
            self._precompute_distance()
            # self._compute_index_pairs()

        self._precompute_subset_of_dists(num_x_and_xt=1_000)
        return self

    def initialize_distance_estimator(
        self,
        num_layers: int,
        num_hidden: int,
        net_type: str = "resnet",
        positive_constraint_fn: str = None,
        net_kwargs: Optional[Dict] = {},
    ):
        """Initialize neural network for distance regression."""
        self.distance_net = DistanceEstimator(
            self.theta,
            self.x,
            self.zscore_distance_precomputed.flatten(),
            num_layers,
            num_hidden,
            net_type,
            positive_constraint_fn,
            **net_kwargs,
        )

    def train(
        self,
        distance_net: Optional[nn.Module] = None,
        training_batch_size: int = 500,
        max_n_epochs: int = 1000,
        stop_after_counter_reaches: int = 50,
        validation_fraction: float = 0.1,
        n_train_per_theta: int = 1,
        n_val_per_theta: int = 1,
        print_every_n: int = 20,
        plot_losses: bool = True,
    ) -> nn.Module:
        # Can use custom distance net, otherwise take existing in class.
        if distance_net != None:
            if self.distance_net != None:
                print("Warning: Overwriting existing distance net.")
            self.distance_net = distance_net

        # Define loss and optimizer.
        nn_loss = nn.MSELoss()
        optimizer = optim.Adam(self.distance_net.parameters())

        # Hold out entire rows of theta/x for validation, but leave all x_targets intact.
        #   The other option is to hold out all thetas, as well as the corresponding xs in x_target.
        dataset = TensorDataset(torch.arange(len(self.theta)))
        train_set, val_set = torch.utils.data.random_split(
            dataset,
            (Tensor([1 - validation_fraction, validation_fraction]) * len(dataset)).to(
                int
            ),
        )
        dataloader = DataLoader(train_set, batch_size=training_batch_size, shuffle=True)

        # Training loop.
        train_losses, val_losses = [], []
        epoch = 0
        self._val_loss = torch.inf
        while epoch <= max_n_epochs and not self._check_convergence(
            epoch, stop_after_counter_reaches
        ):
            time_start = time()

            # If use all validation data, then pre-compute and store all possible indices
            # Otherwise, sample new ones per epoch
            if (epoch == 0) or (n_val_per_theta != -1):
                # Only sample when first epoch, or when not using all validation data
                # i.e., don't resample if using all validation data and epoch > 0
                idx_val = self.make_index_pairs(
                    torch.Tensor(val_set.indices).to(int),
                    torch.arange(len(self.x_target), dtype=int),
                    n_val_per_theta,
                )
                theta_val, _, xt_val, dist_val = self.get_theta_x_distances(idx_val)

            for i_b, idx_theta_batch in enumerate(dataloader):
                optimizer.zero_grad()

                # Randomly sample n x_target for each theta, and get the data.
                idx_batch = self.make_index_pairs(
                    idx_theta_batch[0],
                    torch.arange(len(self.x_target), dtype=int),
                    n_train_per_theta,
                )
                theta_batch, _, xt_batch, dist_batch = self.get_theta_x_distances(
                    idx_batch
                )

                # Forward pass for distances.
                dist_pred = self.distance_net(theta_batch, xt_batch).squeeze()

                # Training loss.
                l = nn_loss(dist_batch, dist_pred)
                l.backward()
                optimizer.step()
                train_losses.append(l.detach())

            # Compute validation loss each epoch.
            with torch.no_grad():
                dist_pred = self.distance_net(theta_val, xt_val).squeeze()
                self._val_loss = nn_loss(dist_val, dist_pred).item()
                val_losses.append([(i_b + 1) * epoch, self._val_loss])

            # Print validation loss
            if epoch % print_every_n == 0:
                print(
                    f"{epoch}: train loss: {l:.6f}, val loss: {self._val_loss:.6f}, {(time()-time_start):.4f} seconds per epoch."
                )

            epoch += 1

        print(f"Network converged after {epoch-1} of {max_n_epochs} epochs.")

        # Plot loss curves for convenience.
        self.train_losses = torch.Tensor(train_losses)
        self.val_losses = torch.Tensor(val_losses)
        if plot_losses:
            self._plot_losses(self.train_losses, self.val_losses)

        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        self.distance_net.zero_grad(set_to_none=True)
        return deepcopy(self.distance_net)

    def predict_distance(self, theta, x):
        # Convenience function that does fixes the shape of x.
        # Expands to have the same batch size as theta, in case x is [1, n_dim].
        if theta.shape[0] != x.shape[0]:
            if len(x.shape) == 2:
                x = x.repeat(theta.shape[0], 1)
            elif len(x.shape) == 3:
                # Has multiple independent observations, i.e., gaussian mixture task.
                x = x.repeat(theta.shape[0], 1, 1)

        with torch.no_grad():
            dist = self.distance_net(theta, x).squeeze(1)
        return dist

    def _check_convergence(self, counter: int, stop_after_counter_reaches: int) -> bool:
        """Return whether the training converged yet and save best model state so far.
        Checks for improvement in validation performance over previous batches or epochs.
        """
        converged = False

        assert self.distance_net is not None
        distance_net = self.distance_net

        # (Re)-start the epoch count with the first epoch or any improvement.
        if counter == 0 or self._val_loss < self._best_val_loss:
            self._best_val_loss = self._val_loss
            self._counts_since_last_improvement = 0
            self._best_model_state_dict = deepcopy(distance_net.state_dict())
        else:
            self._counts_since_last_improvement += 1

        # If no validation improvement over many epochs, stop training.
        if self._counts_since_last_improvement > stop_after_counter_reaches - 1:
            distance_net.load_state_dict(self._best_model_state_dict)
            converged = True

        return converged

    def build_amortized_GLL(self, distance_net: nn.Module = None):
        """Build generalized likelihood function from distance predictor."""
        if distance_net == None:
            distance_net = self.distance_net

        # Build and return function.
        def generalized_loglikelihood(theta: Tensor, x_o: Tensor):
            theta = atleast_2d(theta)
            dist_pred = self.predict_distance(theta, x_o)
            assert dist_pred.shape == (theta.shape[0],)
            return dist_pred

        return generalized_loglikelihood

    def get_potential(self, x_o: Tensor = None, beta: float = 1.0):
        """Make the potential function. Pass through call to GBIPotenial object."""
        return GBIPotential(self.prior, self.build_amortized_GLL(), x_o, beta)

    def build_posterior(
        self, posterior_func: Callable, x_o: Tensor = None, beta: float = 1.0
    ):
        """Create posterior object using the defined potential function."""
        potential_func = self.get_potential(x_o, beta)
        posterior = posterior_func(potential_func, self.prior)
        return posterior

    def _precompute_distance(self):
        """Pre-compute the distances of all pairs of x and x_target."""
        self.distance_precomputed = []
        t_start = time()
        print("Pre-computing distances...", end=" ")
        for x_t in self.x_target:
            self.distance_precomputed.append(
                self.compute_distance(self.x, x_t).unsqueeze(1)
            )
            # self.distance_precomputed.append(
            #     self.distance_func(self.x.unsqueeze(1), x_t).unsqueeze(1)
            # )
        self.distance_precomputed = torch.hstack(self.distance_precomputed)
        print(f"finished in {time()-t_start} seconds.")

    def _precompute_subset_of_dists(self, num_x_and_xt):
        """Pre-compute the distances of some x and x_target pairs for z-scoring."""
        self.zscore_distance_precomputed = []
        random_inds_x = torch.randint(0, len(self.x), (num_x_and_xt,))
        random_inds_xtarget = torch.randint(0, len(self.x_target), (num_x_and_xt,))
        for x_t in self.x_target[random_inds_xtarget]:
            self.zscore_distance_precomputed.append(
                self.compute_distance(self.x[random_inds_x], x_t).unsqueeze(1)
            )
        self.zscore_distance_precomputed = torch.hstack(
            self.zscore_distance_precomputed
        )

    def compute_distance(self, x: Tensor, x_target: Tensor):
        """Compute distance between x and x_target."""
        # x_target should have leading dim of 1 or same as x.
        assert (
            x_target.shape[0] == 1
            or x_target.shape[0] == x.shape[0]
            or len(x_target.shape) == len(x.shape) - 1
        ), f"x_target should have: same leading dim as x, leading dim of 1, or have 1 less dim than x, but have shapes x: {x.shape}, x_target: {x_target.shape}."
        return self.distance_func(x.unsqueeze(1), x_target.unsqueeze(1))

    def subsample_indices(self, n_total, n_samples):
        return torch.randperm(n_total)[:n_samples]

    def make_index_pairs(self, theta_indices, x_target_indices, n_draws_per_theta=1):
        if n_draws_per_theta == -1:
            # Return all possible index pairs.
            return torch.cartesian_prod(theta_indices, x_target_indices)
        else:
            # Return a random subset of index pairs, n per theta.
            index_pairs = []
            for i in range(n_draws_per_theta):
                xt_idx = self.subsample_indices(
                    len(x_target_indices), len(theta_indices)
                )
                index_pairs.append(
                    torch.vstack((theta_indices, x_target_indices[xt_idx])).T
                )
            return torch.concat(index_pairs, dim=0)

    def get_theta_x_distances(self, index_pairs):
        """Return theta, x, x_target, and distance for each index pair."""
        theta = self.theta[index_pairs[:, 0]]
        x = self.x[index_pairs[:, 0]]
        x_target = self.x_target[index_pairs[:, 1]]
        # import pdb; pdb.set_trace()
        if self.do_precompute_distances:
            dist = self.distance_precomputed[index_pairs[:, 0], index_pairs[:, 1]]
        else:
            dist = self.compute_distance(x, x_target)
        return theta, x, x_target, dist

    def _plot_losses(self, train_losses, val_losses):
        plt.figure(figsize=(8, 3))
        plt.plot(train_losses, "k", alpha=0.8)
        plt.plot(val_losses[:, 0], val_losses[:, 1], "r.-", alpha=0.8)
        plt.savefig("losses.png")

    # def _compute_index_pairs(self):
    #     """Return the list of all index pairs for (theta_i, x_target_j)."""
    #     n_lowest = self.n_lowest
    #     n_theta, n_x_target = self.distance_precomputed.shape
    #     if n_lowest == None:
    #         # Not subselecting, return full index list
    #         idx_train = Tensor(np.indices((n_theta, n_x_target)).reshape((2, -1)).T)
    #     else:
    #         # Subselect n lowest distances for train (doesn't work well)
    #         idx_train = torch.vstack(
    #             [
    #                 torch.topk(self.distance_precomputed, n_lowest, 0, largest=False)[
    #                     1
    #                 ].T.reshape(-1),
    #                 torch.arange(n_x_target).repeat((n_lowest, 1)).T.reshape(-1),
    #             ]
    #         ).T
    #     self.idx_train = idx_train.to(int)

    # def _idx_to_data(self, idx_batch):
    #     """Look up for theta, x_target, and distance given indices."""
    #     # Load up the thetas, xs, and distances.
    #     theta_batch, xt_batch = (
    #         self.theta[idx_batch[0][:, 0]],
    #         self.x_target[idx_batch[0][:, 1]],
    #     )
    #     # Look up from precomputed distance matrix.
    #     dist_batch = self.distance_precomputed[idx_batch[0][:, 0], idx_batch[0][:, 1]]

    #     return theta_batch, xt_batch, dist_batch


class GBInferenceEmulator:
    def __init__(
        self,
        emulator_net,
        prior: Distribution,
        distance_func: Callable,
        n_emulator_samples: int = 10,
    ):
        self.distance_func = distance_func
        self.prior = prior
        self.emulator_net = emulator_net
        self.n_emulator_samples = n_emulator_samples

    def build_amortized_GLL(self):
        """Build generalized likelihood function from emulator."""

        # Build and return function.
        def generalized_loglikelihood(theta: Tensor, x_o: Tensor):
            theta = atleast_2d(theta)
            n_iid = x_o.shape[0]
            with torch.no_grad():
                if n_iid == 1:
                    # Single observation, no iid dimension.
                    x_emulator = self.emulator_net.sample(
                        self.n_emulator_samples, theta
                    )
                else:
                    # Multiple iid observations, pad to iid dimension (2).
                    # i.e., (n_theta, n_emulator_samples, iid, data)
                    x_emulator = torch.concat(
                        [
                            self.emulator_net.sample(self.n_emulator_samples, theta)[
                                :, :, None, :
                            ]
                            for _ in range(n_iid)
                        ],
                        2,
                    )

            dist_pred = self.distance_func(x_emulator, x_o)
            assert dist_pred.shape == (theta.shape[0],)
            return dist_pred

        return generalized_loglikelihood

    def get_potential(self, x_o: Tensor = None, beta: float = 1.0):
        """Make the potential function. Pass through call to GBIPotenial object."""
        return GBIPotential(self.prior, self.build_amortized_GLL(), x_o, beta)

    def build_posterior(
        self, posterior_func: Callable, x_o: Tensor = None, beta: float = 1.0
    ):
        """Create posterior object using the defined potential function."""
        potential_func = self.get_potential(x_o, beta)
        posterior = posterior_func(potential_func, self.prior)
        return posterior


class DistanceEstimator(nn.Module):
    def __init__(
        self,
        theta,
        x,
        dists,
        num_layers,
        hidden_features,
        net_type,
        positive_constraint_fn=None,
        dropout_prob=0.0,
        use_batch_norm=False,
        activation="relu",
        activate_output=False,
        trial_net_input_dim=None,
        trial_net_output_dim=None,
        z_score_theta: bool = True,
        z_score_x: bool = True,
        z_score_dists: bool = True,
    ):
        ## TO DO: probably should put all those kwargs in kwargs
        super().__init__()

        theta_dim = theta.shape[1]
        x_dim = x.shape[1]

        if trial_net_input_dim is not None and trial_net_output_dim is not None:
            output_dim_e_net = 20
            trial_net = FCEmbedding(
                input_dim=trial_net_input_dim, output_dim=trial_net_output_dim
            )
            self.embedding_net_x = PermutationInvariantEmbedding(
                trial_net=trial_net,
                trial_net_output_dim=trial_net_output_dim,
                output_dim=output_dim_e_net,
            )
            input_dim = theta_dim + output_dim_e_net
        else:
            self.embedding_net_x = nn.Identity()
            input_dim = theta_dim + x_dim

        if z_score_theta:
            self.z_score_theta_net = standardizing_net(theta, False)
        else:
            self.z_score_theta_net = nn.Identity()

        if z_score_x:
            self.z_score_x_net = standardizing_net(x, False)
        else:
            self.z_score_x_net = nn.Identity()

        if z_score_dists:
            mean_distance = torch.mean(dists)
            std_distance = torch.std(dists)
            self.z_score_dist_net = MultiplyByMean(mean_distance, std_distance)
        else:
            self.z_score_dist_net = nn.Identity()

        output_dim = 1
        if net_type == "MLP":
            net = nets.MLP(
                in_shape=[input_dim],
                out_shape=[output_dim],
                hidden_sizes=[hidden_features] * num_layers,
                activate_output=activate_output,
            )

        elif net_type == "resnet":
            net = nets.ResidualNet(
                in_features=input_dim,
                out_features=output_dim,
                hidden_features=hidden_features,
                num_blocks=num_layers,
                dropout_probability=dropout_prob,
                use_batch_norm=use_batch_norm,
            )
        else:
            raise NotImplementedError

        # ### TO DO: add activation at the end to force positive distance
        if positive_constraint_fn == None:
            self.positive_constraint_fn = lambda x: x
        elif positive_constraint_fn == "relu":
            self.positive_constraint_fn = nn.functional.relu
        elif positive_constraint_fn == "exponential":
            self.positive_constraint_fn = torch.exp
        elif positive_constraint_fn == "softplus":
            self.positive_constraint_fn = nn.functional.softplus
        else:
            raise NotImplementedError

        self.net = net

    def forward(self, theta, x):
        """
        Predicts distance between theta and x.
        """
        # Check for z-score and embedding nets at run time just in case
        if not hasattr(self, "embedding_net_x"):
            self.embedding_net_x = nn.Identity()
        if not hasattr(self, "z_score_theta_net"):
            self.z_score_theta_net = nn.Identity()
        if not hasattr(self, "z_score_x_net"):
            self.z_score_x_net = nn.Identity()
        if not hasattr(self, "z_score_dist_net"):
            self.z_score_dist_net = nn.Identity()
            
        theta = self.z_score_theta_net(theta)
        x = self.z_score_x_net(x)
        x_embedded = self.embedding_net_x(x)
        
        rectified_distance = self.positive_constraint_fn(
            self.net(torch.concat((theta, x_embedded), dim=-1))
        )
        return self.z_score_dist_net(rectified_distance)


class GBIPotential(BasePotential):
    # Need to set this to True for gaussian mixture.
    allow_iid_x = True

    def __init__(self, prior, gen_llh_fn, x_o=None, beta=1.0):
        super().__init__(prior, x_o)
        self.gen_llh_fn = gen_llh_fn
        self.beta = beta

    def set_beta(self, beta):
        self.beta = beta

    def __call__(self, theta, track_gradients=True):
        with torch.set_grad_enabled(track_gradients):
            return -self.beta * self.gen_llh_fn(theta, self.x_o) + self.prior.log_prob(
                theta
            )
