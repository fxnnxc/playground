import torch 

def construct_rho(**rule_description):
    if "gamma" in rule_description:
        g = rule_description['gamma']
        def _gamma_fn(w,b):
            w = w + w * torch.max(torch.tensor(0, device=w.device), w) * g
            if b is not None : 
                b = b+ b * torch.max(torch.tensor(0, device=b.device), b) * g 
            return w,b 
        return _gamma_fn
    else:
        return lambda w,b : (w, b)


def construct_incr(**rule_description):
    if "epsilon" in rule_description:
        e = rule_description['epsilon']
        return lambda x : x + e
    else:
        return lambda x : x
