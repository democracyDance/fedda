import torch.nn as nn

def adversarial_loss(domain_preds, domain_labels):
    criterion = nn.BCELoss()
    return criterion(domain_preds, domain_labels)
