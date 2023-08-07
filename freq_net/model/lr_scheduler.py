from torch.optim.lr_scheduler import CosineAnnealingLR


def cossine_lr(optimizer, t_max, eta_min=1e-7):
    """
    eta_max will be the learning rate of optimizer
    """
    return CosineAnnealingLR(optimizer=optimizer, T_max=t_max, eta_min=eta_min)
