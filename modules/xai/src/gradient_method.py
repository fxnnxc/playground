
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


def integrated_gradient(model, x, baseline, M, to=torch.device('cuda')):

    def make_interpolation(x, base, M):
        lst = [] 
        for i in range(M):
            alpha = i/M
            interpolated =x * (alpha) + base * (1-alpha)
            lst.append(interpolated.clone())
        return torch.stack(lst)

    model.to(to)

    X = make_interpolation(x, baseline, M)
    X = Variable(X, requires_grad=True).to(to)
    X.retain_grad()
    x = x.to(to)
    baseline = baseline.to(to)
    
    output = model.forward(X)
    output.sum().backward()

    gradient = X.grad

    IG = (x-baseline) * gradient.sum(axis=0)
    return IG, X, gradient

def vanila_gradient(model, x, to=torch.device('cuda')):
    model.to(to)
    X = x.unsqueeze(0)
    X = Variable(X, requires_grad=True).to(to)
    X.retain_grad()
    
    output = model.forward(X)
    output.sum().backward()

    vanila_gradient = X.grad

    return vanila_gradient


def smooth_gradient(model, x,  M, sigma, to=torch.device('cuda')):
    def make_perturbation(x, M, sigma=1):
        lst = [] 
        for i in range(M):
            noise = torch.normal(0, sigma, size=x.size())
            lst.append(x.clone() + noise.clone())
        return torch.stack(lst)
    
    model.to(to)
    X = make_perturbation(x, M, sigma)
    X = Variable(X, requires_grad=True).to(to)
    X.retain_grad()    
    
    output = model.forward(X)

    output.sum().backward()


    vanila_gradient = X.grad

    return vanila_gradient



    
