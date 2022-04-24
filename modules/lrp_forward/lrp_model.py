import torch
from torch import nn
from copy import deepcopy
from deeping.xai.lrp.lrp_layers import *
"""
based on https://github.com/KaiFabi/PyTorchRelevancePropagation
"""
class LRPModel(nn.Module):
    def __init__(self,layers:torch.nn.ModuleList, 
                    kwargs_list:list, 
                    ignore_mean_std = True,
                    mean=torch.Tensor([0.485, 0.456, 0.406]).reshape(1,-1,1,1),
                    std=torch.Tensor([0.229, 0.224, 0.225]).reshape(1,-1,1,1)) -> None:

        super().__init__()
        self.ignore_maen_std = ignore_mean_std
        self.layers = LRPModel.make_ModuleList(layers)
        self.kwargs_list = kwargs_list        
        if len(self.kwargs_list) < len(self.layers):
            self.kwargs_list = [self.kwargs_list[0]] * len(self.layers)
        elif len(self.kwargs_list) > len(self.layers):
            self.kwargs_list = self.kwargs_list[:len(self.layers)]


        self.lrp_layers = self._create_lrp_model()
        self.mean = mean 
        self.std = std
        self.activation = nn.ReLU()
    
    def cuda(self):
        super().cuda()
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        
    def _create_lrp_model(self) -> torch.nn.ModuleList:
        layers = deepcopy(self.layers)
        # Run backwards through layers
        for i, (layer, kwargs) in enumerate(zip(layers[::-1], self.kwargs_list[::-1])):
            try:
                layers[i] = ModuleLRPTable[layer.__class__](layer=layer, **kwargs)
            except KeyError:
                message = f"Layer-wise relevance propagation not implemented for " \
                          f"{layer.__class__.__name__} layer."
                raise NotImplementedError(message)
        return layers

    @staticmethod
    def make_ModuleList(layer_list):
        layers = torch.nn.ModuleList()
        for layer in layer_list:
            layers.append(layer)
        return layers

    def forward(self, x: torch.tensor) -> torch.tensor:
        activations = list()
        # Run inference and collect activations.
        with torch.no_grad():
            # Replace image with ones avoids using image information for relevance computation.
            activations.append(torch.ones_like(x))
            for i, layer in enumerate(self.layers):
                x = layer.forward(x)
                if i != len(self.layers) -1 :
                    x = self.activation(x)
                activations.append(x)

        # Reverse order of activations to run backwards through model
        activations = activations[::-1]
        activations = [a.data.requires_grad_(True) for a in activations]

        # Initial relevance scores are the network's output activations
        relevance = torch.softmax(activations.pop(0), dim=-1)  # Unsupervised
        all_relevance = [relevance]

        # Perform relevance propagation
        for i, layer in enumerate(self.lrp_layers):
            if i < len(self.lrp_layers)-1 or self.ignore_maen_std:
                relevance = layer.forward(activations.pop(0), relevance)
            else:
                a = activations.pop(0)
                lb = (a.data*0+(0-self.mean)/self.std).requires_grad_(True)
                hb = (a.data*0+(1-self.mean)/self.std).requires_grad_(True)

                z = layer.layer.forward(a) + 1e-9                                     # step 1 (a)
                z -= apply_rule_to_layer(layer.layer, rule="fn", function=lambda p: p.clamp(min=0)).forward(lb)    # step 1 (b)
                z -= apply_rule_to_layer(layer.layer, rule="fn", function=lambda p: p.clamp(max=0)).forward(hb)    # step 1 (c)
                s = (relevance/z).data                                                      # step 2
                (z*s).sum().backward()
                c,cp,cm = a.grad,lb.grad,hb.grad            # step 3
                relevance = (a*c + lb*cp + hb*cm).data   

            all_relevance.append(relevance)

        # output = relevance.permute(0, 2, 3, 1).sum(dim=-1).squeeze().detach().cpu()
        output = relevance
        return output, all_relevance[::-1]

    def get_layer_info(self):
        return [layer.get_info() for layer in self.lrp_layers]

if __name__ == "__main__":
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(20,10)
            self.linear2 = nn.Linear(10,5)
        def forward(self, x):
            x = self.linear1(x) 
            x = torch.nn.functional.relu(x)
            x = self.linear2(x)
            x = torch.nn.functional.relu(x)
            return x 

    # --- test fcn ---
    x = torch.rand(size=(1, 20))
    model = TestModel()
    layers = [model.linear1, model.linear2]
    kwargs_list = [{'eps':1e-4, "rule":"gamma", "gamma":0.5}, {'eps':1e-4, "rule":"z_plus"}]

    lrp_model = LRPModel(layers, kwargs_list, mean=torch.rand(1), std=torch.rand(1))
    r, all_relevance = lrp_model.forward(x)
    print(r.size())
    print(lrp_model.get_layer_info())
    
    # --- test vgg ---
    import torchvision
    x = torch.rand(size=(1, 3, 224, 224))
    model = torchvision.models.vgg16(pretrained=True)
    layers = []
    for layer in model.features:
        layers.append(layer)

    layers.append(model.avgpool)
    layers.append(torch.nn.Flatten(start_dim=1))

    for layer in model.classifier:
        layers.append(layer)


    lrp_model = LRPModel(layers, kwargs_list=[{'eps':1e-4, "rule":"z_plus"}]* len(layers))
    r, all_relevance = lrp_model.forward(x)
    print(r.size())

    print(lrp_model.get_layer_info())
    