import torch
import pickle


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
