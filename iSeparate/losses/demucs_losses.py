import torch
import torch.nn as nn


def recon_sum_equal_mix_loss(output, target):
    loss1 = l1_loss(output, target)
    loss2 = l1_loss(output.sum(1, keepdim=True), target.sum(1, keepdim=True))
    return loss1 + loss2


def l1_loss(output, target, weights=None):
    if weights is None:
        weights = [1.0, 1.0, 1.0, 1.0]
    weights = torch.tensor(weights).to(output)
    reduction_dims = tuple(range(2, output.dim()))
    loss = nn.L1Loss(reduction="none")(output, target)
    loss = loss.mean(dim=reduction_dims).mean(0)
    loss = (loss * weights).sum() / weights.sum()
    return loss


def l2_loss(output, target, weights=None):
    if weights is None:
        weights = [1.0, 1.0, 1.0, 1.0]
    weights = torch.tensor(weights).to(output)
    reduction_dims = tuple(range(2, output.dim()))
    loss = nn.MSELoss(reduction="none")(output, target)
    loss = loss.mean(dim=reduction_dims).mean(0)
    loss = (loss * weights).sum() / weights.sum()
    return loss

def huber_loss(output, target, weights=None, delta=1.0):
    """
    delta: Threshold at which delta-scaled L1 and L2 losses interchange
           Value must be non-negative.
           default: 1.0
    """
    if weights is None:
        weights = [1.0, 1.0, 1.0, 1.0]
    weights = torch.tensor(weights).to(output)
    reduction_dims = tuple(range(2, output.dim()))
    loss = nn.HuberLoss(reduction="none", delta=delta)(output, target)
    loss = loss.mean(dim=reduction_dims).mean(0)
    loss = (loss * weights).sum() / weights.sum()
    return loss

def smooth_loss(output, target, weights=None, beta=1.0):
    """
    beta: Threshold at which L1 and L2 losses interchange
          Value must be non-negative.
          default: 1.0
    """
    if weights is None:
        weights = [1.0, 1.0, 1.0, 1.0]
    weights = torch.tensor(weights).to(output)
    reduction_dims = tuple(range(2, output.dim()))
    loss = nn.SmoothL1Loss(reduction="none", beta=beta)(output, target)
    loss = loss.mean(dim=reduction_dims).mean(0)
    loss = (loss * weights).sum() / weights.sum()
    return loss
