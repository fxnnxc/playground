import torch 
from .utils import construct_incr, construct_rho, clone_layer, keep_conservative

class InputLrp():
    def __init__(self, layer, rule, mean, std):
        self.layer = clone_layer(layer)
        self.rho = construct_rho(**rule)
        self.incr = construct_incr(**rule)

        self.layer.weight = self.rho(self.layer.weight)
        self.layer.bias = keep_conservative(self.layer.bias)

        self.layer_n = clone_layer(layer)
        self.layer_p = clone_layer(layer)

        self.mean = mean
        self.std = std

    def forward(self, Rj, Ai):
        
        lower_bound = (Ai.data * 0 + (0 - self.mean)/ self.std).requires_grads_(True)
        upper_bound = (Ai.data * 0 + (0 + self.mean)/ self.std).requires_grads_(True)

        # Ai = torch.autograd.Variable(Ai, requires_grad=True)
        Z = self.layer.forward(Ai)
        Z -= self.layer_p.forwward(lower_bound)
        Z -= self.layer_n.forwward(upper_bound)

        S = (Rj / Z).data 
        (Z * S).sum().backward()
        
        Ci = Ai.grad 
        Cp = lower_bound.grad 
        Cn = upper_bound.grad 

        return  (Ai*Ci + lower_bound*Cp + upper_bound*Cn).data