import numpy as np
from torch.optim import lr_scheduler

warm_restart_epochs = [0, 10, 30, 70, 150, 310]

def _cosine_annealing(step, lr_max, lr_min):
    for i, epoch in enumerate(warm_restart_epochs):
        if step >= epoch:
            update_cycle_len = warm_restart_epochs[i+1] - warm_restart_epochs[i-1]
            step -= warm_restart_epochs[i]
            break

    new_lr = lr_min + (lr_max - lr_min) * 0.5 * (
             1 + np.cos(step / update_cycle_len * np.pi))
    return new_lr


def cosine_annealing_scheduler(optimizer, lr):
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_annealing(
            step=step,
            lr_max=lr,  # since lr_lambda computes multiplicative factor
            lr_min=0.001))

    return scheduler

