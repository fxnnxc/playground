import torch 
import torch.nn as nn 

class DropoutLrp(nn.Dropout):
    def __init__(self, layer, rule):
        self.layer = layer

    def forward(self, Rj, Ai):
        return Rj



