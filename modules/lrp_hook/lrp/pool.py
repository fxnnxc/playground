import torch 
import torch.nn as nn 

class MaxPoolLrp(nn.MaxPool2d):
    def forward(self, input, **kwargs):
        return super().forward(input)
    
    @classmethod 
    def from_torch(cls, max_pool):
        module = cls(max_pool.kernel_size)        
        return module


class AvgPoolLrp(nn.AvgPool2d):
    def forward(self, input, **kwargs):
        return super().forward(input)
    
    @classmethod 
    def from_torch(cls, avg_pool):
        module = cls(avg_pool.kernel_size)        
        return module
