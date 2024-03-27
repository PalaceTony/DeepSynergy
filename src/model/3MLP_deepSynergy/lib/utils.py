import numpy as np
import torch


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def set_seed(seed_value):
    np.random.seed(seed_value)  # Set numpy seed
    torch.manual_seed(seed_value)  # Set pytorch seed CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)  # Set pytorch seed GPU
        torch.cuda.manual_seed_all(seed_value)  # for multiGPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
