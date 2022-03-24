import torch
from torch import nn
from copy import deepcopy
from lrp_layers import *

ModuleLRPTable = {
    torch.nn.modules.linear.Linear: RelevancePropagationLinear,
    torch.nn.modules.conv.Conv2d: RelevancePropagationConv2d,
    torch.nn.modules.activation.ReLU: RelevancePropagationReLU,
    torch.nn.modules.dropout.Dropout: RelevancePropagationDropout,
    torch.nn.modules.flatten.Flatten: RelevancePropagationFlatten,
    torch.nn.modules.pooling.AvgPool2d: RelevancePropagationAvgPool2d,
    torch.nn.modules.pooling.MaxPool2d: RelevancePropagationMaxPool2d,
    torch.nn.modules.pooling.AdaptiveAvgPool2d: RelevancePropagationAdaptiveAvgPool2d
}

class LRPModel(nn.Module):
 
    def __init__(self, model: torch.nn.Module, custom_layer_path:torch.nn.ModuleList=None) -> None:
        super().__init__()
        self.model = model
        self.model.eval()  # self.model.train() activates dropout / batch normalization etc.!
        self.layers = custom_layer_path if custom_layer_path is not None else self._get_layer_operations()
        self.lrp_layers = self._create_lrp_model()

    def _create_lrp_model(self) -> torch.nn.ModuleList:
        layers = deepcopy(self.layers)
        # Run backwards through layers
        for i, layer in enumerate(layers[::-1]):
            try:
                layers[i] = ModuleLRPTable[layer.__class__](layer=layer)
            except KeyError:
                message = f"Layer-wise relevance propagation not implemented for " \
                          f"{layer.__class__.__name__} layer."
                raise NotImplementedError(message)
        return layers

    @staticmethod
    def make_custom_layer_path(layer_list):
        layers = torch.nn.ModuleList()
        for layer in layer_list:
            layers.append(layer)
        return layers


    def _get_layer_operations(self) -> torch.nn.ModuleList:
        layers = torch.nn.ModuleList()
        for layer in list(self.model.modules()):
            if layer.__class__  in ModuleLRPTable.keys():
                layers.append(layer)
  
        return layers

    def forward(self, x: torch.tensor) -> torch.tensor:
        activations = list()
        # Run inference and collect activations.
        with torch.no_grad():
            # Replace image with ones avoids using image information for relevance computation.
            activations.append(torch.ones_like(x))
            for layer in self.layers:
                x = layer.forward(x)
                activations.append(x)

        # Reverse order of activations to run backwards through model
        activations = activations[::-1]
        activations = [a.data.requires_grad_(True) for a in activations]

        # Initial relevance scores are the network's output activations
        relevance = torch.softmax(activations.pop(0), dim=-1)  # Unsupervised
        all_relevance = [relevance]

        # Perform relevance propagation
        for i, layer in enumerate(self.lrp_layers):
            relevance = layer.forward(activations.pop(0), relevance)
            all_relevance.append(relevance)

        # output = relevance.permute(0, 2, 3, 1).sum(dim=-1).squeeze().detach().cpu()
        output = relevance
        return output, all_relevance[::-1]

    
import matplotlib.pyplot as plt

def plot_relevance_scores(x: torch.tensor, r: torch.tensor, name: str, config: dict) -> None:
    """Plots results from layer-wise relevance propagation next to original image.
    Method currently accepts only a batch size of one.
    Args:
        x: Original image.
        r: Relevance scores for original image.
        name: Image name.
        config: Dictionary holding configuration.
    Returns:
        None.
    """
    output_dir = config["output_dir"]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    x = x[0].squeeze().permute(1, 2, 0).detach().cpu()
    x_min = x.min()
    x_max = x.max()
    x = (x - x_min) / (x_max - x_min)
    axes[0].imshow(x)
    axes[0].set_axis_off()

    r_min = r.min()
    r_max = r.max()
    r = (r - r_min) / (r_max - r_min)
    axes[1].imshow(r, cmap="afmhot")
    axes[1].set_axis_off()

    fig.tight_layout()
    plt.savefig(f"{output_dir}/image_{name}.png")
    plt.close(fig)


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
    lrp_model = LRPModel(model)
    r, all_relevance = lrp_model.forward(x)
    print(r.size())

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

    custom_layer_path = LRPModel.make_custom_layer_path(layers)
    lrp_model = LRPModel(model, custom_layer_path=custom_layer_path)
    r, all_relevance = lrp_model.forward(x)
    print(r.size())

    