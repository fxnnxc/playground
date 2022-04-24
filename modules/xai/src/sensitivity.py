
import torch 
import torch.nn as nn
def sensitivity(model, x,  to="cpu"):
    model.to(to)
    X = torch.autograd.Variable(x.unsqueeze(0), requires_grad=True).to(to)
    X.retain_grad()

    y_hat = model(X)
    y_hat.mean().backward()

    return (X.grad)**2 