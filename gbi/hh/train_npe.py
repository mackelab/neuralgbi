import pickle
from sbi.inference import SNPE
import gbi.hh.utils as utils
from sbi.utils import posterior_nn
import torch

from hydra.utils import get_original_cwd
import hydra
from omegaconf import DictConfig
import logging

log = logging.getLogger("hh_npe")


@hydra.main(version_base="1.1", config_path="config", config_name="npe")
def train_npe(cfg: DictConfig) -> None:
    path = get_original_cwd()

    _ = torch.manual_seed(5555)

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

    theta = theta[: cfg.nsims]
    x = x[: cfg.nsims]

    true_params, labels_params = utils.obs_params(reduced_model=False)
    prior = utils.prior(
        true_params=true_params,
        prior_uniform=True,
        prior_extent=True,
        prior_log=False,
        seed=0,
    )

    density_estimator = posterior_nn(
        cfg.density_estimator, prior=prior, sigmoid_theta=cfg.sigmoid_theta
    )
    inference = SNPE(prior=prior, density_estimator=density_estimator)

    _ = inference.append_simulations(theta, x).train(
        training_batch_size=cfg.training_batch_size
    )

    with open("inference_npe.pkl", "wb") as handle:
        pickle.dump(inference, handle)


if __name__ == "__main__":
    train_npe()
