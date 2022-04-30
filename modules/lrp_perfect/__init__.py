
import torch
import torch.nn as nn 
from .modules import *

LookUpTable = {
    "Input" : InputLrp,
    "Linear" : LinearLrp,
    "ReLU" : ReluLrp,
    "Conv2d" : Conv2dLrp,
    "MaxPool2d": AvgPoolLrp,  # treat Max pool as Avg Pooling
    "AvgPool2d" : AvgPoolLrp,
    "Flatten":FlattenLrp,
    "Dropout" : DropoutLrp,
}

class LRP():
    def __init__(self, layers, rule_descriptions, device,  mean=None, std=None):
        super().__init__()
        self.rule_description = rule_descriptions
        self.original_layers = layers
        self.mean = mean 
        self.std = std
        self.lrp_modules = self.construct_lrp_modules(self.original_layers, rule_descriptions, device)
    
        assert len(layers) == len(rule_descriptions)


    def forward(self, a):
        # store activations 
        activations = [a] 
        for i, layer in enumerate(self.original_layers):
            a = layer(a)
            activations.append(a)
        
        activations = activations[::-1]
        activations = [a.data.requires_grad_(True) for a in activations]

        # compute LRP 
        relevances = [torch.softmax(activations.pop(0), dim=-1)] 
        for (Ai, module) in zip(activations, self.lrp_modules):
            Rj = relevances[-1]
            Ri = module.forward(Rj, Ai)
            relevances.append(Ri)

        return relevances, activations 

    def construct_lrp_modules(self, original_layers, rule_descriptions, device):
        used_names = [] 
        modules = [] 

        for i, layer in enumerate(original_layers):
            rule = rule_descriptions[i]
            if i==0 and self.mean is not None:
                lrp_module = LookUpTable["Input"](layer, rule, device,  self.mean, self.std)
                lrp_module.layer.to(device)
                lrp_module.layer_n.to(device)
                lrp_module.layer_p.to(device)
            else:
                name  = layer.__class__.__name__
                assert name in LookUpTable
                lrp_module = LookUpTable[name](layer, rule)
                lrp_module.layer.to(device)
            modules.append(lrp_module)
            used_names.append(name)
        
        self.kind_warning(used_names)
        return modules[::-1]

    def kind_warning(self, used_names):
        if "ReLU" not in used_names:
            print(f'[Kind Warning] : ReLU is not in the layers. You should manually add activations.' )
            print(f'[Kind Warning] : Are you sure your model structure excludes ReLU : <{used_names}>?')

