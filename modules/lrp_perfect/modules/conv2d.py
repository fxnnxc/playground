import torch 
from .utils import construct_incr, construct_rho, clone_layer, keep_conservative

class Conv2dLrp():
    def __init__(self, layer, rule, device):
        self.device = device 
        self.layer = clone_layer(layer).to(self.device)
        self.rho = construct_rho(**rule)
        self.incr = construct_incr(**rule)

        self.layer.weight = self.rho(self.layer.weight)
        self.layer.bias = keep_conservative(self.layer.bias)

    def forward(self, Rj, Ai):
        
        Ai = torch.autograd.Variable(Ai, requires_grad=True).to(self.device)
        Z = self.layer.forward(Ai)
        S = (Rj / Z).data 
        (Z * S).sum().backward()
        Ci = Ai.grad 

        return  (Ai * Ci).data


