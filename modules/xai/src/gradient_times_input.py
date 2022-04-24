import torch 

def gradient_times_input(model, x, to="cpu"):
    model.to(to)
    X = torch.autograd.Variable(x.unsqueeze(0), requires_grad=True).to(to)
    X.retain_grad()

    y_hat = model(X)
    y_hat.mean().backward()

    return X.detach() * X.grad.detach()