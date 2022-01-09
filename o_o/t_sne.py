import torch 

def compute_kernel(k, X,Y, get_pairs=False):
    # compute kernel for all pairs in X and Y 
    assert X.ndim==2 and  Y.ndim==2
    assert X.size(1) == Y.size(1)
    v = 0 
    if not get_pairs:    
        for i in range(X.size(0)):
            for j in range(Y.size(0)):
                v += k(X[i,:], Y[i,:])
        return v
    else:
        pairs = []
        for i in range(X.size(0)):
            pairs.append([])
            for j in range(Y.size(0)):
                pair_value = k(X[i,:], Y[i,:])
                pairs[-1].append(pair_value)
                v += pair_value
            pairs[-1] = torch.stack(pairs[-1])
        return v, torch.stack(pairs)


def rbf_kernel(x,y, sigma=1):
    v = (x-y)**2 / (2 * sigma**2)
    return torch.exp(v)

def entropy_loss(P, Q):
    # compute the entropy
    return 0


def sne(X, kernel, dim=2, iteration=10):
    v, P = compute_kernel(kernel, X, X, get_pairs=True)    
    # optimize 
    Y = torch.rand(X.size(0), dim)
    for i in range(iteration):
        v2, Q = compute_kernel(kernel, Y, Y, get_pairs=True)
        loss = entropy_loss(P, Q)
        
    return Y
    

if __name__ == "__main__":
    X = torch.rand(5,2)
    Y = torch.rand(7,2)
    print(compute_kernel(rbf_kernel, X, Y,  get_pairs=False))
    print(compute_kernel(rbf_kernel, X, Y,  get_pairs=True))