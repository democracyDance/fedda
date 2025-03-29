import torch.nn.functional as F

def consistency_loss(pred1, pred2):
    return F.mse_loss(pred1, pred2)
