import copy
import torch

def federated_average(models):
    global_model = copy.deepcopy(models[0])
    for key in global_model.state_dict().keys():
        if global_model.state_dict()[key].dtype in [torch.float32, torch.float64, torch.float16]:
            avg = torch.stack([model.state_dict()[key].float() for model in models], dim=0).mean(dim=0)
            global_model.state_dict()[key].copy_(avg)
        else:
            # 对于 int 类型参数 (如 num_batches_tracked)，取第一个模型的值
            global_model.state_dict()[key].copy_(models[0].state_dict()[key])
    return global_model


