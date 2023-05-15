import pickle
import torch
from torch import Tensor

from gbi.GBI import GBInference
import gbi.hh.utils as utils
import numpy as np

from hydra.utils import get_original_cwd
import hydra
from omegaconf import DictConfig
import logging

log = logging.getLogger("hh_gbi")


class MaeDistZ:
    def __init__(self, std):
        self.std = std

    def __call__(self, xs: Tensor, x_o: Tensor) -> Tensor:
        # Shape of xs should be [num_thetas, num_xs, num_x_dims].
        dist = (xs - x_o).abs()
        dist /= self.std
        mae = dist.mean(dim=2)  # Average over data dimensions.
        return mae.mean(dim=1)  # Monte-Carlo average


@hydra.main(version_base="1.1", config_path="config", config_name="gbi")
def train_gbi(cfg: DictConfig) -> None:
    """Train GBI"""
    path = get_original_cwd()
    _ = torch.manual_seed(42)

    if cfg.type == "allen":
        with open(
            f"{path}/../../results/hh/simulations/allen_theta.pkl", "rb"
        ) as handle:
            theta = pickle.load(handle)

        with open(
            f"{path}/../../results/hh/simulations/allen_summstats.pkl", "rb"
        ) as handle:
            x = pickle.load(handle)
    elif cfg.type == "synthetic":
        with open(f"{path}/data/theta.pkl", "rb") as handle:
            theta = pickle.load(handle)

        with open(f"{path}/data/summstats.pkl", "rb") as handle:
            x = pickle.load(handle)
    else:
        raise NameError

    log.info(f"num sims loaded: theta {len(theta)}, x {len(x)}")

    obs_stats_ls, _ = utils.load_all_allen()
    obs_stats_ls = torch.as_tensor(np.concatenate(obs_stats_ls), dtype=torch.float32)

    theta = theta[: cfg.nsims]
    x = x[: cfg.nsims]
    data_std = torch.std(x, dim=0)

    mae_dist_z = MaeDistZ(data_std)
    n_nonaug_x = cfg.nsims
    n_augmented_x = cfg.nsims if cfg.n_augmented_x is None else cfg.n_augmented_x

    x_target_condition = x[:, 0] > 5.0
    x_t = x[x_target_condition]
    x_target_condition = x_t[:, 0] < 40.0
    x_t = x_t[x_target_condition]

    x_aug = x_t[torch.randint(x_t.shape[0], size=(n_augmented_x,))]
    x_aug = x_aug + torch.randn(x_aug.shape) * x_t.std(dim=0) * cfg.noise_level
    print("num nonaug: ", x_t[:n_nonaug_x].shape)
    print("num x_aug: ", x_aug.shape)
    x_target = torch.cat([x_t[:n_nonaug_x], x_aug])

    if cfg.deal_with_xo == "do_not_use":
        x_target = x_target
    elif cfg.deal_with_xo == "append":
        x_target = torch.cat(
            [obs_stats_ls[: cfg.n_obs_for_target].repeat((1000, 1)), x_target]
        )
    elif cfg.deal_with_xo == "use":
        x_target = obs_stats_ls[: cfg.n_obs_for_target].repeat((200_000, 1))
    else:
        raise NameError

    true_params, labels_params = utils.obs_params(reduced_model=False)

    prior = utils.prior(
        true_params=true_params,
        prior_uniform=True,
        prior_extent=True,
        prior_log=False,
        seed=0,
    )

    inference = GBInference(
        prior, mae_dist_z, do_precompute_distances=cfg.do_precompute_distances
    )
    inference = inference.append_simulations(theta, x, x_target)
    inference.initialize_distance_estimator(
        num_layers=cfg.num_layers,
        num_hidden=cfg.num_hidden,
        net_type=cfg.net_type,
        positive_constraint_fn=cfg.positive_constraint_fn,
        net_kwargs={
            "z_score_theta": True,
            "z_score_x": True,
            "z_score_dists": True,
        },
    )

    _ = inference.train(
        training_batch_size=cfg.training_batch_size,
        max_n_epochs=cfg.max_epochs,
        stop_after_counter_reaches=cfg.n_epochs_convergence,
        print_every_n=cfg.print_every_n,
        plot_losses=False,
        validation_fraction=cfg.validation_fraction,
        n_train_per_theta=cfg.n_train_per_theta,
        n_val_per_theta=cfg.n_val_per_theta,
    )

    best_validation_loss = inference._best_val_loss
    log.info(f"best val loss: {best_validation_loss}")

    with open("inference_gbi.pkl", "wb") as handle:
        pickle.dump(inference, handle)


if __name__ == "__main__":
    train_gbi()
