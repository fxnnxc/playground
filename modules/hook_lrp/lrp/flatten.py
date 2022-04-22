import torch 
import torch.nn as nn

class FlattenLrp(nn.Flatten):
    def forward(self, input, **kwargs):
        return super().forward(input)
    
    @classmethod 
    def from_torch(cls, flatten):
        module = cls(flatten.start_dim, flatten.end_dim)        
        return module