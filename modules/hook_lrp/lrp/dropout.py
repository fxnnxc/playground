import torch 
import torch.nn as nn 

class DropoutLrp(nn.Dropout):
    def forward(self, input, **kwargs):
        return super().forward(input)
    
    @classmethod 
    def from_torch(cls, dropout):
        module = cls(dropout.p)        
        return module
