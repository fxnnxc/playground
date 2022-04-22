import torch 
import torch.nn.functional as F
from torch.autograd import Function
from .utils import construct_incr, construct_rho


def _forward_rho(ctx, input, weight, bias, rho, incr):
    ctx.save_for_backward(input, weight, bias)
    ctx.rho = rho 
    ctx.incr = incr
    return F.linear(input, weight, bias)

def _backward_rho(ctx, relevance_output):
    input, weight, bias = ctx.saved_tensors 
    rho                 = ctx.rho 
    incr                = ctx.incr

    weight, bias = rho(weight, bias)
    Z = incr(F.linear(input,  weight, bias))   
    
    relevance_output = relevance_output / Z 
    relevance_input = F.linear(relevance_output, weight.t(), bias=None)
    relevance_input = relevance_input * input 
 
    return relevance_input, None, None, None, None


class LinearLrpFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, rho, incr, bias=None):
        return _forward_rho(ctx, input, weight, bias, rho, incr)

    @staticmethod
    def backward(ctx, relevance_output):
        return _backward_rho(ctx, relevance_output)


class LinearLrp(torch.nn.Linear):
    def forward(self, input, **kwargs):
        rho = construct_rho(**kwargs)
        incr = construct_incr(**kwargs)
        return LinearLrpFunction.apply(input, self.weight, rho, incr, self.bias)
    
    @classmethod 
    def from_torch(cls, lin):
        bias = lin.bias is not None 
        module = cls(lin.in_features, lin.out_features, bias)
        module.load_state_dict(lin.state_dict())
        
        return module



