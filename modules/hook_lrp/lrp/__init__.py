
import torch
import torch.nn as nn 
from .linear import LinearLrp
from .activation import ReluLrp
from .conv2d import Conv2dLrp
from .pool import MaxPoolLrp, AvgPoolLrp
from .flatten import FlattenLrp
from .dropout import DropoutLrp

LookUpTable = {
    "Linear" : LinearLrp,
    "ReLU" : ReluLrp,
    "Conv2d" : Conv2dLrp,
    "MaxPool2d": MaxPoolLrp,
    "AvgPool2d" : AvgPoolLrp,
    "Flatten":FlattenLrp,
    "Dropout" : DropoutLrp,
}

class LrpModel(nn.Module):
    def __init__(self, layers, rule_description):
        super().__init__()
        self.rule_description = rule_description
        self.original_layers = layers
        self.modules = self.construct_lrp_modules(self.original_layers)

    def forward(self, x):
        for i, module in enumerate(self.modules):
            x = module(x, **(self.rule_description[i] if self.rule_description[i] else {}))
        return x 

    def construct_lrp_modules(self, original_layers):
        used_names = [] 

        modules = [] 
        for layer in original_layers:
            name  = layer.__class__.__name__
            assert name in LookUpTable
            lrp_module = LookUpTable[name].from_torch(layer)
            modules.append(lrp_module)
            used_names.append(name)
        
        self.kind_warning(used_names)
        return modules 

    def kind_warning(self, used_names):
        if "ReLU" not in used_names:
            print(f'[Kind Warning] : ReLU is not in the layers. You should manually add activations.' )
            print(f'[Kind Warning] : Are you sure your model structure excludes ReLU : <{used_names}>?')

