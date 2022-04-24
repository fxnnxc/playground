import torch 
from torch import nn
from deeping.xai.lrp.lrp_utils import *

class RelevancePropavationBase(nn.Module):
    def __init__(self, layer, rule=None, *args, **kwargs):
        super().__init__()
        self.rule = rule
        self.layer = layer
        self.args = args 
        self.kwargs = kwargs

    def get_info(self):
        if len(type(self).__name__) <1:
            print(self)
        return (type(self).__name__, self.rule, self.args, self.kwargs)

class RelevancePropagationAdaptiveAvgPool2d(RelevancePropavationBase):
    def __init__(self, layer: torch.nn.AdaptiveAvgPool2d, rule,  eps, top_k_percent=1.0, **kwargs) -> None:
        super().__init__(layer, rule,  eps, top_k_percent, **kwargs)
        self.layer = apply_rule_to_layer(layer, rule, **kwargs)
        self.eps = eps
        self.top_k_percent = top_k_percent

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r

class RelevancePropagationAvgPool2d(RelevancePropavationBase):
    def __init__(self, layer: torch.nn.AvgPool2d, rule, eps,  **kwargs) -> None:
        super().__init__(layer, rule, eps,  **kwargs)
        self.layer = apply_rule_to_layer(layer, rule, **kwargs)
        self.eps = eps

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r


class RelevancePropagationMaxPool2d(RelevancePropavationBase):
    def __init__(self, layer: torch.nn.MaxPool2d, rule, eps, mode: str = "avg",  **kwargs) -> None:
        super().__init__(layer, rule, eps, mode,  **kwargs)
        if mode == "avg":
            self.layer = torch.nn.AvgPool2d(kernel_size=(2, 2))
        elif mode == "max":
            self.layer = layer

        self.layer = apply_rule_to_layer(self.layer, rule, **kwargs)
        self.eps = eps

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r


class RelevancePropagationConv2d(RelevancePropavationBase):
    def __init__(self, layer: torch.nn.Conv2d, rule, eps: float, top_k_percent=1.0, **kwargs) -> None:
        super().__init__(layer, rule, eps, top_k_percent, **kwargs)
        self.top_k_percent = top_k_percent
        self.layer = apply_rule_to_layer(self.layer, rule, **kwargs)
        self.eps = eps

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        r = relevance_filter(r, top_k_percent=self.top_k_percent )
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r


class RelevancePropagationLinear(RelevancePropavationBase):
    def __init__(self, layer: torch.nn.Linear, rule, eps: float = 1.0e-05, top_k_percent=1.0, **kwargs) -> None:
        super().__init__(layer, rule, eps, top_k_percent, **kwargs)
        self.layer = apply_rule_to_layer(self.layer, rule , **kwargs)
        self.top_k_percent= top_k_percent
        self.eps = eps

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        r = relevance_filter(r, top_k_percent=self.top_k_percent)
        z = self.layer.forward(a) + self.eps
        s = r / z
        c = torch.mm(s, self.layer.weight)
        r = (a * c).data
        return r


class RelevancePropagationFlatten(RelevancePropavationBase):
    def __init__(self, layer: torch.nn.Flatten,  **kwargs) -> None:
        super().__init__(layer,  **kwargs)

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        r = r.view(size=a.shape)
        return r


class RelevancePropagationReLU(RelevancePropavationBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r


class RelevancePropagationDropout(RelevancePropavationBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r


class RelevancePropagationIdentity(RelevancePropavationBase):
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r


ModuleLRPTable = {
    torch.nn.modules.linear.Linear: RelevancePropagationLinear,
    torch.nn.modules.conv.Conv2d: RelevancePropagationConv2d,
    torch.nn.modules.activation.ReLU: RelevancePropagationReLU,
    torch.nn.modules.dropout.Dropout: RelevancePropagationDropout,
    torch.nn.modules.flatten.Flatten: RelevancePropagationFlatten,
    torch.nn.modules.pooling.AvgPool2d: RelevancePropagationAvgPool2d,
    torch.nn.modules.pooling.MaxPool2d: RelevancePropagationMaxPool2d,
    torch.nn.modules.pooling.AdaptiveAvgPool2d: RelevancePropagationAdaptiveAvgPool2d,
}