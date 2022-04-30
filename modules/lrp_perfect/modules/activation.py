import torch 
import torch.nn as nn 

class ReluLrp():
    def __init__(self, layer, rule):
        self.layer = layer

    def forward(self, Rj, Ai):
        return Rj


