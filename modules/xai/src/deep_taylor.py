

import torch 
import torch.nn as nn 
import copy 

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(20, 10)
        self.linear2 = nn.Linear(10, 1)
        self.linear2.weight.data = torch.ones(1, 10)
        self.linear2.weight.require_grad = False

    def forward(self, x_i):
        x_j = self.linear1(x_i)
        x_j = nn.functional.relu(x_j)
        x_k = self.linear2(x_j)
        return x_k

    def compute_lrp(self, x_i):
        # store activation 
        activations = [x_i]
        x_j = nn.functional.relu(self.linear1(x_i))
        activations.append(x_j)
        x_k = self.linear2(x_j)
        activations.append(x_k)

        # store computed relevances 
        relevances = [x_k, x_j]
        
        R_j = x_j # (1 x 10)
        A_i = activations[0] # (1 x 20)
        W_ij = self.linear1.weight  # (10 x 20) 

        Z = torch.mm(W_ij, A_i.t()) # (1 x 20) x (20 x 10)  = (1 x 10)
        C_i = torch.mm(R_j/Z, W_ij ) # (1 x 10) x (10 x 20) = (1 x 20) 
        R_i = A_i * C_i       #  (1 x 20 ) x (1 x 20)
        
        relevances.append(R_i)

        l = copy.deepcopy(self.linear1)
        l.bias = nn.Parameter(0 * l.bias)
        l.weight = nn.Parameter(self.linear1.weight)
        R_j = x_j 
        A_i = activations[0]
        A_i = torch.autograd.Variable(A_i, requires_grad=True)

        Z = l.forward(A_i)
        S = (R_j / Z).data 
        (Z * S).sum().backward() # summation of all relevance in R

        C_i = A_i.grad 
        R_i = A_i * C_i
        print("Difference ")
        print((relevances[-1] - R_i).sum())

        return relevances, activations

if __name__ == "__main__":
    model = Model()
    x_i = torch.rand(1, 20)

    # function value
    r_k = model(x_i)
    R, A = model.compute_lrp(x_i)
    # relevance
    # r_j = 

    # relevance