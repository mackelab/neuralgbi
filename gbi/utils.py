import torch


def concatenate_xs(x1, x2):
    return torch.concat([x1, x2], dim=0)


## TO DO: function for adding noise to x to expand "observations" to tackle misspecification
def add_noise_to_x():
    return
