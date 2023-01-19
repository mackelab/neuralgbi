import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.utils.data import TensorDataset, DataLoader

from sbi.utils.torchutils import atleast_2d
from sbi.inference.potentials.base_potential import BasePotential

from pyknos.nflows.nn import nets


class GBInference:
    def __init__(
        self, prior, distance_func, do_precompute_distances=True, n_lowest=None
    ):
        self.prior = prior
        self.distance_func = distance_func
        self.do_precompute_distances = do_precompute_distances
        self.n_lowest = n_lowest

    def append_simulations(self, theta, x, x_target):
        self.theta = theta
        self.x = x
        self.x_target = x_target
        if self.do_precompute_distances:
            self._precompute_distance()
            self._compute_index_pairs()
        return self

    def initialize_distance_estimator(
        self, num_layers, num_hidden, net_type, net_kwargs={}
    ):
        self.distance_net = DistanceEstimator(
            self.theta.shape[1],
            self.x.shape[1],
            num_layers,
            num_hidden,
            net_type,
            **net_kwargs,
        )

    def train(
        self,
        distance_net=None,
        training_batch_size=100,
        n_epochs=100,
        validation_fraction=0.1,
        print_every_n=10,
        plot_losses=True,
    ):

        ## TO DO:
        ## TAKE IS_CONVERGED FROM SBI

        # Can use custome distance net, otherwise take existing in class.
        if distance_net == None:
            distance_net = self.distance_net
        else:
            self.distance_net = distance_net

        idx_train, theta, x_target, dist_precomputed = (
            self.idx_train,
            self.theta,
            self.x_target,
            self.distance_precomputed,
        )
        # Define loss and optimizer
        nn_loss = nn.MSELoss()
        optimizer = optim.Adam(distance_net.parameters())

        # Splitting train and validation set
        dataset = TensorDataset(idx_train)
        train_set, val_set = torch.utils.data.random_split(
            dataset,
            (Tensor([1 - validation_fraction, validation_fraction]) * len(dataset)).to(
                int
            ),
        )
        dataloader = DataLoader(train_set, batch_size=training_batch_size, shuffle=True)

        # Get validation set once
        theta_val, x_val, dist_val = self._idx_to_data(val_set[:])

        train_losses, val_losses = [], []
        for e in range(n_epochs):
            for i_b, idx_batch in enumerate(dataloader):
                optimizer.zero_grad()

                # load batch of theta, x, and pre-computed distances
                theta_batch, x_batch, dist_batch = self._idx_to_data(idx_batch)                

                # forward pass for distances
                dist_pred = distance_net(theta_batch, x_batch).squeeze()

                # training loss
                l = nn_loss(dist_batch, dist_pred)
                l.backward()
                optimizer.step()
                train_losses.append(l.detach())

            # compute validation loss
            dist_pred = distance_net(theta_val, x_val).squeeze()
            l_val = nn_loss(dist_val, dist_pred)
            val_losses.append([i_b * e, l_val.detach()])

            if e % print_every_n == 0:
                print(f"{e}: {l}")

        train_losses = torch.Tensor(train_losses)
        val_losses = torch.Tensor(val_losses)
        if plot_losses:
            self._plot_losses(train_losses, val_losses)
        return distance_net

    def build_amortized_GLL(self, distance_net: nn.Module = None):
        if distance_net == None:
            distance_net = self.distance_net
        # Build and return function
        def generalized_loglikelihood(theta: Tensor, x_o: Tensor):
            theta = atleast_2d(theta)
            dist_pred = distance_net(theta, x_o.repeat((theta.shape[0], 1))).squeeze(1)
            assert dist_pred.shape == (theta.shape[0],)
            return dist_pred

        return generalized_loglikelihood

    def get_potential(self, x_o=None, beta=1.0):
        return GBIPotential(self.prior, self.build_amortized_GLL(), x_o, beta)

    def build_posterior(self, posterior_func, x_o=None, beta=1.0):
        potential_func = self.get_potential(x_o, beta)
        posterior = posterior_func(potential_func, self.prior)
        return posterior

    def _precompute_distance(self):
        self.distance_precomputed = torch.hstack(
            [
                self.distance_func(self.x.unsqueeze(1), x_t).unsqueeze(1)
                for x_t in self.x_target
            ]
        )

    def _compute_index_pairs(self):
        n_lowest = self.n_lowest
        n_theta, n_x_target = self.distance_precomputed.shape
        if n_lowest == None:
            # Not subselecting, return full index list
            idx_train = Tensor(np.indices((n_theta, n_x_target)).reshape((2, -1)).T)
        else:
            # Subselect n lowest distances for train (doesn't work well)
            idx_train = torch.vstack(
                [
                    torch.topk(self.distance_precomputed, n_lowest, 0, largest=False)[
                        1
                    ].T.reshape(-1),
                    torch.arange(n_x_target).repeat((n_lowest, 1)).T.reshape(-1),
                ]
            ).T
        self.idx_train = idx_train.to(int)

    def _plot_losses(self, train_losses, val_losses):
        plt.plot(train_losses, "k", alpha=0.8)
        plt.plot(val_losses[:, 0], val_losses[:, 1], "ro-", alpha=0.8)

    def _idx_to_data(self, idx_batch):
        # Load up the thetas, xs, and distances
        theta_batch, x_batch = (
            self.theta[idx_batch[0][:, 0]],
            self.x_target[idx_batch[0][:, 1]],
        )

        # Look up from precomputed distance matrix
        dist_batch = self.distance_precomputed[idx_batch[0][:, 0], idx_batch[0][:, 1]]

        return theta_batch, x_batch, dist_batch


class DistanceEstimator(nn.Module):
    def __init__(
        self,
        theta_dim,
        x_dim,
        num_layers,
        hidden_features,
        net_type="MLP",
        dropout_prob=0.0,
        use_batch_norm=False,
        activation="relu",
        activate_output=False,
    ):
        ## TO DO: probably should put all those kwargs in kwargs
        super().__init__()
        input_dim = theta_dim + x_dim
        output_dim = 1
        if net_type == "MLP":
            self.net = nets.MLP(
                in_shape=[input_dim],
                out_shape=[output_dim],
                hidden_sizes=[hidden_features] * num_layers,
                activate_output=activate_output,
            )

        elif net_type == "resnet":
            self.net = nets.ResidualNet(
                in_features=input_dim,
                out_features=output_dim,
                hidden_features=hidden_features,
                num_blocks=num_layers,
                dropout_probability=dropout_prob,
                use_batch_norm=use_batch_norm,
            )
        ### TO DO: add activation at the end to force positive distance

    def forward(self, theta, x):
        """
        Predicts distance between theta and x
        """
        return self.net(torch.concat((theta, x), dim=-1))


class GBIPotential(BasePotential):
    allow_iid_x = False

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
