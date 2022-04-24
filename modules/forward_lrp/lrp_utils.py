import torch 
import torch.nn as nn 
import copy 

rho = lambda p, gamma: p + gamma * p.clamp(min=0)

def apply_rule_to_layer(layer, rule, **kwargs):
    layer = copy.deepcopy(layer)
    if rule == "z_plus":
        if hasattr(layer, "weight"):
            layer.weight = torch.nn.Parameter(layer.weight.clamp(min=0.0))
        if hasattr(layer, "bias"):
            layer.bias = torch.nn.Parameter(torch.zeros_like(layer.bias))
    elif rule == "gamma":
        if hasattr(layer, "weight"):
            gamma = kwargs.get("gamma")
            layer.weight = torch.nn.Parameter(rho(layer.weight, gamma))
            layer.bias = torch.nn.Parameter(rho(layer.bias, gamma))
    elif rule =="zero":
        if hasattr(layer, "weight"):
            pass
    elif rule =="fn":
        if hasattr(layer, "weight"):
            function = kwargs.get("function")
            try: layer.weight = nn.Parameter(function(layer.weight))
            except AttributeError: pass
            try: layer.bias   = nn.Parameter(function(layer.bias))
            except AttributeError: pass

    return layer


def relevance_filter(r: torch.tensor, top_k_percent: float = 1.0) -> torch.tensor:
    assert 0.0 <= top_k_percent <= 1.0
    if top_k_percent < 1.0:
        size = r.size()
        r = r.flatten(start_dim=1)
        num_elements = r.size(-1)
        k = int(top_k_percent * num_elements)
        assert k > 0, f"Expected k to be larger than 0."
        top_k = torch.topk(input=r, k=k, dim=-1)
        r = torch.zeros_like(r)
        r.scatter_(dim=1, index=top_k.indices, src=top_k.values)
        return r.view(size)
    else:
        return r
