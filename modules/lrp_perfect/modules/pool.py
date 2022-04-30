import torch 
from .utils import construct_incr, clone_layer, keep_conservative

class AvgPoolLrp():
    def __init__(self, layer, rule):
        rule = {k: v for k,v in rule.items() if k=="epsilon"}  # only epsilont rule is possible
        self.layer = clone_layer(layer)
        self.incr = construct_incr(**rule)

    def forward(self, Rj, Ai):
        
        # Ai = torch.autograd.Variable(Ai, requires_grad=True)
        Z = self.layer.forward(Ai)
        Z = self.incr(Z)
        S = (Rj / Z).data 
        (Z * S).sum().backward()
        Ci = Ai.grad 

        return  (Ai * Ci).data
