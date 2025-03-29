# utils/fedadg.py

def fedadg(models):
    # 同样为 placeholder
    from fedalgorithm.federated_averaging import federated_average
    return federated_average(models)
