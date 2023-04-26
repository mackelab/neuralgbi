import torch
import pickle
from gbi.distances import mmd_dist


def concatenate_xs(x1, x2):
    return torch.concat([x1, x2], dim=0)


def pickle_dump(full_path, data_dump):
    with open(full_path, "wb") as handle:
        pickle.dump(data_dump, handle)


def pickle_load(filename):
    with open(filename, "rb") as f:
        loaded = pickle.load(f)
    return loaded


## TO DO: function for adding noise to x to expand "observations" to tackle misspecification
def add_noise_to_x():
    return


def compute_predictives(theta_samples, task, distance_fn_empirical, inference_obj=None):
    predictives = {}
    predictives["theta"] = theta_samples
    # simulate posterior predictives
    predictives["x_pred"] = task.simulate(theta_samples)

    # compute posterior predictive distances
    # true average distance
    # check if it's mmd, if so, need to do loop
    if distance_fn_empirical == mmd_dist:
        predictives["dist_gt"] = torch.Tensor(
            [task.distance_fn(th) for th in theta_samples]
        )
    else:
        predictives["dist_gt"] = task.distance_fn(theta_samples)

    # sample-based estimate
    predictives["dist_samples"] = distance_fn_empirical(
        predictives["x_pred"][:, None, :], task.x_o[None, None, :]
    )

    if inference_obj:
        with torch.no_grad():
            # predictives["dist_estimate"] = inference_obj.distance_net(
            #     theta_samples, task.x_o.repeat((theta_samples.shape[0], 1))
            # ).squeeze(1)
            predictives["dist_estimate"] = inference_obj.predict_distance(
                theta_samples, task.x_o[None, :]
            )
    else:
        predictives["dist_estimate"] = None
    return predictives


def compute_moments(predictives):
    summaries = {}
    for k, v in predictives.items():
        if "dist" in k:
            summaries[f"{k}_mean"], summaries[f"{k}_std"] = (
                v.mean(0).numpy(),
                v.std(0).numpy(),
            )
    # Compute correlation and error between true and estimated distances
    summaries["r_dist_gt_estimate"] = torch.corrcoef(
        torch.vstack((predictives["dist_gt"], predictives["dist_estimate"]))
    )[0, 1].numpy()
    summaries["mse_dist_gt_estimate"] = (
        ((predictives["dist_gt"] - predictives["dist_estimate"]) ** 2).mean().numpy()
    )

    summaries["r_dist_samples_estimate"] = torch.corrcoef(
        torch.vstack((predictives["dist_samples"], predictives["dist_estimate"]))
    )[0, 1].numpy()
    summaries["mse_dist_samples_estimate"] = (
        ((predictives["dist_samples"] - predictives["dist_estimate"]) ** 2)
        .mean()
        .numpy()
    )
    return summaries
