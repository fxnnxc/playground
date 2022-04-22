import torch 
import torch.nn as nn 

class ReluLrp(nn.ReLU):
    def forward(self, input, **kwargs):
        return super().forward(input)
    
    @classmethod 
    def from_torch(cls, relu):
        module = cls()        
        return module


class TanhLrp(nn.Tanh):
    def forward(self, input, **kwargs):
        raise NotImplementedError("I'm not sure yet...")
        return super().forward(input)
    
    @classmethod 
    def from_torch(cls, tanh):
        module = cls()        
        return module
