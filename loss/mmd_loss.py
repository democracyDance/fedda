import torch
import torch.nn as nn

def compute_mmd(source_features, target_features):
    source_mean = torch.mean(source_features, dim=0)
    target_mean = torch.mean(target_features, dim=0)
    mmd_loss = torch.norm(source_mean - target_mean, p=2)
    return mmd_loss
