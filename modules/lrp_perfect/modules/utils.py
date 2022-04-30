import torch
import torch.nn as nn 
import copy 

def construct_rho(**rule_description):
    if "gamma" in rule_description:
        g = rule_description['gamma']
        def _gamma_fn(w):
            w = w + w * torch.max(torch.tensor(0, device=w.device), w) * g
            return nn.Parameter(w)
        return _gamma_fn
    else:
        return lambda w : w

def keep_conservative(b): 
    # set bias to 0
    return nn.Parameter(0 * b)

def construct_incr(**rule_description):
    if "epsilon" in rule_description:
        e = rule_description['epsilon']
        return lambda x : x + e
    else:
        return lambda x : x


def clone_layer(layer):
    cloned_layer = copy.deepcopy(layer)
    if hasattr(layer, "weight"):
        cloned_layer.weight = nn.Parameter(layer.weight)
    if hasattr(layer, "bias"):
        cloned_layer.bias = nn.Parameter(layer.bias)
    return cloned_layer