import numpy as np
from torch.optim import lr_scheduler


def _cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
        1 + np.cos(step / total_steps * np.pi))


def cosine_annealing_scheduler(optimizer, epochs, lr):
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_annealing(
            step,
            epochs,
            lr,  # since lr_lambda computes multiplicative factor
            0))

    return scheduler