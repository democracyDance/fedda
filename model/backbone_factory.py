# model/backbone_factory.py

import torch.nn as nn
import torchvision.models as models

def get_feature_extractor(name="resnet50"):
    if name == "resnet50":
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]
        model = nn.Sequential(*modules, nn.Flatten(), nn.Linear(2048, 256))
        return model
    elif name == "vit":
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model = nn.Sequential(vit, nn.Linear(1000, 256))
        return model
    else:
        raise NotImplementedError(f"Backbone {name} not supported.")
