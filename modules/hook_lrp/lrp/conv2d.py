import torch 
import torch.nn.functional as F
from torch.autograd import Function
from .utils import construct_incr, construct_rho


def _forward_rho(ctx, input, weight, bias, stride, padding, dilation, groups, rho, incr):
    ctx.save_for_backward(input, weight, bias)
    ctx.rho = rho 
    ctx.incr = incr
    ctx.stride = stride
    ctx.padding = padding
    ctx.dilation = dilation
    ctx.groups = groups

    return  F.conv2d(input, weight, bias, stride, padding, dilation, groups)

def _backward_rho(ctx, relevance_output):
    input, weight, bias = ctx.saved_tensors 
    rho                 = ctx.rho 
    incr                = ctx.incr

    weight, bias = rho(weight, bias)
    Z = incr(F.conv2d(input, weight, bias, ctx.stride, ctx.padding, ctx.dilation, ctx.groups))   
    
    relevance_output = relevance_output / Z 
    relevance_input = F.conv_transpose2d(relevance_output, weight, None, padding=ctx.padding)
    relevance_input = relevance_input * input 
 
    return relevance_input, None, None, None, None, None, None, None, None, None


class Conv2dLrpFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, rho, incr, bias=None, stride=1, padding=0, dilation=1, groups=1, **kwargs):
        return _forward_rho(ctx, input, weight, bias, stride, padding, dilation, groups, rho, incr)

    @staticmethod
    def backward(ctx, relevance_output):
        return _backward_rho(ctx, relevance_output)


class Conv2dLrp(torch.nn.Conv2d):
    def forward(self, input, **kwargs):
        rho = construct_rho(**kwargs)
        incr = construct_incr(**kwargs)
        return Conv2dLrpFunction.apply(input, self.weight, rho, incr, self.bias)
    
    @classmethod 
    def from_torch(cls, conv):
        in_channels = conv.weight.shape[1] * conv.groups
        bias = conv.bias is not None 

        module = cls(in_channels, 
                    conv.out_channels, 
                    conv.kernel_size, 
                    conv.stride, 
                    conv.padding, 
                    conv.dilation, 
                    conv.groups,
                    bias=bias, 
                    padding_mode=conv.padding_mode
                )
        module.load_state_dict(conv.state_dict())
        return module



