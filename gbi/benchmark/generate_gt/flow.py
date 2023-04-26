from functools import partial
from typing import Optional
import pickle
from copy import deepcopy

import torch
from torch.utils import data
from pyknos.nflows import distributions as distributions_
from pyknos.nflows import flows, transforms
from pyknos.nflows.nn import nets
from torch import Tensor, nn, relu, tensor, uint8, optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data.sampler import SubsetRandomSampler

from sbi.utils.sbiutils import standardizing_transform, z_score_parser
from sbi.utils.torchutils import create_alternating_binary_mask


def train_flow(training_batch_size: int = 500):
    with open("mcmc_samples.pkl", "rb") as handle:
        samples = pickle.load(handle)

    net = build_nsf(
        samples,
        z_score_x="independent",
        hidden_features=50,
        num_transforms=5,
        num_bins=10,
    )
    trained_nn = train(net, samples, training_batch_size=training_batch_size)
    with open("trained_nn.pkl", "wb") as handle:
        pickle.dump(trained_nn, handle)


def converged(
    net,
    epoch: int,
    stop_after_epochs: int,
    _val_log_prob,
    _best_val_log_prob,
    _epochs_since_last_improvement,
    _best_model_state_dict,
):
    """Return whether the training converged yet and save best model state so far.

    Checks for improvement in validation performance over previous epochs.

    Args:
        epoch: Current epoch in training.
        stop_after_epochs: How many fruitless epochs to let pass before stopping.

    Returns:
        Whether the training has stopped improving, i.e. has converged.
    """
    converged = False

    assert net is not None
    neural_net = net

    # (Re)-start the epoch count with the first epoch or any improvement.
    if epoch == 0 or _val_log_prob > _best_val_log_prob:
        _best_val_log_prob = _val_log_prob
        _epochs_since_last_improvement = 0
        _best_model_state_dict = deepcopy(neural_net.state_dict())
    else:
        _epochs_since_last_improvement += 1

    # If no validation improvement over many epochs, stop training.
    if _epochs_since_last_improvement > stop_after_epochs - 1:
        neural_net.load_state_dict(_best_model_state_dict)
        converged = True

    return (
        converged,
        stop_after_epochs,
        _val_log_prob,
        _best_val_log_prob,
        _epochs_since_last_improvement,
        _best_model_state_dict,
    )


def train(
    net,
    theta,
    device="cpu",
    training_batch_size: int = 500,
    learning_rate: float = 5e-4,
    validation_fraction: float = 0.1,
    stop_after_epochs: int = 20,
    max_num_epochs: int = 2 ** 31 - 1,
    clip_max_norm: Optional[float] = 5.0,
):
    dataset = data.TensorDataset(theta)

    # Get total number of training examples.
    num_examples = theta.size(0)
    # Select random train and validation splits from (theta, x) pairs.
    num_training_examples = int((1 - validation_fraction) * num_examples)
    num_validation_examples = num_examples - num_training_examples

    # Seperate indicies for training and validation
    permuted_indices = torch.randperm(num_examples)
    train_indices, val_indices = (
        permuted_indices[:num_training_examples],
        permuted_indices[num_training_examples:],
    )

    # Create training and validation loaders using a subset sampler.
    # Intentionally use dicts to define the default dataloader args
    # Then, use dataloader_kwargs to override (or add to) any of these defaults
    # https://stackoverflow.com/questions/44784577/in-method-call-args-how-to-override-keyword-argument-of-unpacked-dict
    train_loader_kwargs = {
        "batch_size": min(training_batch_size, num_training_examples),
        "drop_last": True,
        "sampler": SubsetRandomSampler(train_indices.tolist()),
    }
    val_loader_kwargs = {
        "batch_size": min(training_batch_size, num_validation_examples),
        "shuffle": False,
        "drop_last": True,
        "sampler": SubsetRandomSampler(val_indices.tolist()),
    }
    train_loader = data.DataLoader(dataset, **train_loader_kwargs)
    val_loader = data.DataLoader(dataset, **val_loader_kwargs)

    # Move entire net to device for training.
    net = net.to(device)

    optimizer = optim.Adam(list(net.parameters()), lr=learning_rate)
    epoch, _val_log_prob, _best_val_log_prob = 0, float("-Inf"), float("-Inf")
    _epochs_since_last_improvement = 0
    _best_model_state_dict = None

    while epoch <= max_num_epochs:
        (
            conv,
            stop_after_epochs,
            _val_log_prob,
            _best_val_log_prob,
            _epochs_since_last_improvement,
            _best_model_state_dict,
        ) = converged(
            net,
            epoch,
            stop_after_epochs,
            _val_log_prob,
            _best_val_log_prob,
            _epochs_since_last_improvement,
            _best_model_state_dict,
        )
        if conv:
            break

        # Train for a single epoch.
        net.train()
        train_log_probs_sum = 0
        for batch in train_loader:
            optimizer.zero_grad()
            # Get batches on current device.
            theta_batch = batch[0].to(device)

            train_losses = -net.log_prob(theta_batch)
            train_loss = torch.mean(train_losses)
            train_log_probs_sum -= train_losses.sum().item()

            train_loss.backward()
            if clip_max_norm is not None:
                clip_grad_norm_(net.parameters(), max_norm=clip_max_norm)
            optimizer.step()

        epoch += 1

        # Calculate validation performance.
        net.eval()
        val_log_prob_sum = 0

        with torch.no_grad():
            for batch in val_loader:
                theta_batch = batch[0].to(device)
                # Take negative loss here to get validation log_prob.
                val_losses = -net.log_prob(theta_batch)
                val_log_prob_sum -= val_losses.sum().item()

        # Take mean over all validation samples.
        _val_log_prob = val_log_prob_sum / (
            len(val_loader) * val_loader.batch_size  # type: ignore
        )
        print("epoch", epoch)

    # Avoid keeping the gradients in the resulting network, which can
    # cause memory leakage when benchmarking.
    net.zero_grad(set_to_none=True)

    return deepcopy(net)


def build_nsf(
    batch_x: Tensor,
    z_score_x: Optional[str] = "independent",
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_bins: int = 10,
    tail_bound: float = 3.0,
    num_blocks: int = 2,
    dropout_probability: float = 0.0,
    use_batch_norm: bool = False,
    **kwargs,
) -> nn.Module:
    """Builds NSF p(x|y).

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network, can be one of:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        z_score_y: Whether to z-score ys passing into the network, same options as
            z_score_x.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms.
        num_bins: Number of bins used for the splines.
        embedding_net: Optional embedding network for y.
        tail_bound: tail bound for each spline.
        hidden_layers_spline_context: number of hidden layers of the spline context net
            for one-dimensional x.
        num_blocks: number of blocks used for residual net for context embedding.
        dropout_probability: dropout probability for regularization in residual net.
        use_batch_norm: whether to use batch norm in residual net.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for maf and are therefore ignored.

    Returns:
        Neural network.
    """
    x_numel = batch_x[0].numel()

    # Define mask function to alternate between predicted x-dimensions.
    def mask_in_layer(i):
        return create_alternating_binary_mask(features=x_numel, even=(i % 2 == 0))

    # If x is just a scalar then use a dummy mask and learn spline parameters using the
    # conditioning variables only.
    if x_numel == 1:
        raise NotImplementedError
    else:
        # Use conditional resnet as spline conditioner.
        conditioner = partial(
            nets.ResidualNet,
            hidden_features=hidden_features,
            context_features=None,
            num_blocks=num_blocks,
            activation=relu,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

    # Stack spline transforms.
    transform_list = []
    for i in range(num_transforms):
        block = [
            transforms.PiecewiseRationalQuadraticCouplingTransform(
                mask=mask_in_layer(i) if x_numel > 1 else tensor([1], dtype=uint8),
                transform_net_create_fn=conditioner,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
                apply_unconditional_transform=False,
            )
        ]
        # Add LU transform only for high D x. Permutation makes sense only for more than
        # one feature.
        if x_numel > 1:
            block.append(transforms.LULinear(x_numel, identity_init=True))
        transform_list += block

    z_score_x_bool, structured_x = z_score_parser(z_score_x)
    if z_score_x_bool:
        # Prepend standardizing transform to nsf transforms.
        transform_list = [
            standardizing_transform(batch_x, structured_x)
        ] + transform_list

    distribution = distributions_.StandardNormal((x_numel,))

    # Combine transforms.
    transform = transforms.CompositeTransform(transform_list)
    neural_net = flows.Flow(transform, distribution, None)

    return neural_net


if __name__ == "__main__":
    train_flow()
