# based on the code in https://github.com/dfdazac/wassdistance

import torch 
import torch.nn as nn 

class SinkhornDistance(nn.Module):
    def __init__(self, eps, max_iter, reduction="none", p=1):
        super().__init__()
        self.eps = eps 
        self.max_iter =max_iter 
        self.reduction = reduction
        self.p = p
        
    def forward(self, x,y):
        C = self._cost_matrix(x,y,self.p)
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2: 
            batch_size = 1 
        else:
            batch_size = x.shape[0]
            
        # --- 
        mu = torch.empty(batch_size, x_points, dtype=torch.float, requires_grad=False).fill_(1.0/x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float, requires_grad=False).fill_(1.0/y_points).squeeze()
        
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        
        thresh = 1e-3    # stopping 
        
        # --- iteartion 
        for i in range(self.max_iter):
            u1 = u 
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u 
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, 1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()
            
            if err.item() < thresh:
                break 
            
        U, V = u,v 
        pi = torch.exp(self.M(C, U, V))
        # --- Sinkhorn distance 
        cost = torch.sum(pi * C, dim=(-2, -1))
        if self.reduction == "mean":
            cost = cost.mean() 
        elif self.reduction == "sum":
            cost = cost.sum()
                    
        return (cost)**(1/self.p), pi, C
            
        
    def M(self, C, u, v):
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps
        
    
    @staticmethod 
    def _cost_matrix(x,y,p=1):
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum(torch.abs(x_col - y_lin)** p, -1)
        return C
    
    @staticmethod
    def ave(u, u1, tau):
        return tau * u + (1-tau) * u1
    
    
import matplotlib.pyplot as plt
import numpy as np 
def plot_shinkhorn_result(P, C):
    fig, ax = plt.subplots(1,2, figsize=(8,4))
    a1 = ax[0].imshow(C)
    ax[0].set_title('Distance matrix')
    a2 = ax[1].imshow(P)
    ax[1].set_title('Coupling matrix')
    
    fig.colorbar(a1, ax=ax[0])
    fig.colorbar(a2, ax=ax[1])
    
    
    
def show_assignments(a, b, P, arrow=False):    
    plt.scatter(a[:, 0], a[:, 1])
    plt.scatter(b[:, 0], b[:, 1])
    plt.legend(["a", "b"])

    norm_P = P/P.max()
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            if arrow:
                plt.arrow(a[i, 0], a[i, 1], b[j, 0]-a[i, 0], b[j, 1]-a[i, 1],
                        alpha=norm_P[i,j].item(),head_width=0.03, length_includes_head=True)
            else:
                plt.arrow(a[i, 0], a[i, 1], b[j, 0]-a[i, 0], b[j, 1]-a[i, 1],
                     alpha=norm_P[i,j].item())
    plt.title('Assignments')

    plt.axis('off')
    
    

def samples_to_cdf(a, b):
    # assume that a and b are 1D samples 
    cdf_a = range(0,100)
    cdf_b = range(0,100)