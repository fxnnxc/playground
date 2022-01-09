import torch 
import matplotlib.pyplot as plt 
from torch.autograd import Variable

def compute_kernel(k, X,Y):
    # compute kernel for all pairs in X and Y 
    assert X.ndim==2 and  Y.ndim==2
    assert X.size(1) == Y.size(1)

    pairs = []
    for i in range(X.size(0)):
        pairs.append([])
        for j in range(Y.size(0)):
            pair_value = k(X[i,:], Y[j,:])
            pairs[-1].append(pair_value)
        pairs[-1] = torch.stack(pairs[-1])
    return torch.stack(pairs)

def rbf_kernel(x,y, sigma=10):
    v = ((x-y).norm(2))**2 / (2 * sigma**2)
    return torch.exp(v)

def entropy_loss(P, Q):
    # compute the entropy
    loss = P * P.log() -  P * Q.log()
    return loss.sum()


def sne(X, kernel, dim=2, iteration=5, learning_rate=0.1):
    P = compute_kernel(kernel, X, X)   
    P /= P.clone().detach().norm(2, dim=0)
    # optimize 
    Y = Variable(torch.rand(X.size(0), dim), requires_grad=True)
    for i in range(iteration):
        Q = compute_kernel(kernel, Y, Y)
        Q /= Q.clone().detach().norm(2, dim=0)
        loss = entropy_loss(P, Q)
        if loss <0:
            break
        loss.backward()
        Y.data -= learning_rate * Y.grad.data
    return Y
    

def plot_2d(Y):
    plt.scatter(Y[:,0], Y[:,1])
    

if __name__ == "__main__":
    X = torch.rand(100,20)
    X[:,2:20] = 0
    Y = sne(X, rbf_kernel)
    plot_2d(Y.detach().numpy())