import numpy as np
import torch
import logging
import datetime
import os


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


def configure_logging(args):
    # Generate output directory with the current timestamp
    now = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    output_dir = os.path.join(args.output_dir, args.model, f"{now}")
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir

    log_format = f"%(asctime)s - {args.model} - %(message)s"
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)

    # Remove all existing handlers
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])

    # File handler for output log file
    file_handler = logging.FileHandler(os.path.join(output_dir, "training.log"))
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)

    # Stream handler for console output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    return logger
