# utils/fedprox.py

def fedprox(models, mu=0.01):
    # 这里只是 placeholder，你可根据 FedProx 论文具体实现
    # 当前默认返回普通 FedAvg
    from fedalgorithm.federated_averaging import federated_average
    return federated_average(models)
