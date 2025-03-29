# utils/aggregation_factory.py
#目前main中的聚合函数统一写死用fedavg，factory只是预留位
from fedalgorithm.federated_averaging import federated_average
from fedalgorithm.fedprox import fedprox
from fedalgorithm.fedadg import fedadg

def get_aggregation(name="fedavg"):
    if name == "fedavg":
        return federated_average
    elif name == "fedprox":
        return fedprox
    elif name == "fedadg":
        return fedadg
    else:
        raise NotImplementedError(f"Aggregation {name} not supported.")
