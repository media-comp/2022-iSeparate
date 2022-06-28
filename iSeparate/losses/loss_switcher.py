import torch

from iSeparate.losses.demucs_losses import l1_loss, l2_loss, recon_sum_equal_mix_loss


def get_loss_fn(loss_name):
    if loss_name == "demucs_l1":
        return l1_loss
    elif loss_name == "demucs_l2":
        return l2_loss
    elif loss_name == "demucs_huber":
        return huber_loss
    elif loss_name == "demucs_smoothl1":
        return smoothl1_loss
    elif loss_name == "demucs_recon_sum_equal_mix_loss":
        return recon_sum_equal_mix_loss
    elif loss_name == "l1_loss":
        return torch.nn.L1Loss()
    elif loss_name == "l2_loss":
        return torch.nn.MSELoss()
    elif loss_name == "huber_loss":
        return torch.nn.HuberLoss()
    elif loss_name == "smoothl1_loss":
        return torch.nn.SmoothL1Loss()
