import torch 
import numpy as np

def power_method(weight, power_iteration=1):
    assert weight.ndim == 2
    dim_in, dim_out = weight.size(1), weight.size(0)    
    u = torch.rand(dim_out)
    v = torch.rand(dim_in)
    
    for i in range(power_iteration):
        u = torch.matmul(weight, v) / torch.matmul(weight, v).norm()
        assert u.size(0) == dim_out
        v = torch.matmul(weight.T, u) / torch.matmul(weight.T, u).norm()
        assert v.size(0) == dim_in
        
    return torch.matmul(torch.matmul(weight, v).T, u)
    


if __name__ == "__main__":
    power_iteration = 2
    dim_in, dim_out = 2, 10
    weight = torch.rand(dim_out, dim_in)
    largest_singular_value = power_method(weight, power_iteration=power_iteration)
    numpy_result =  np.linalg.svd(weight.numpy())[1][0]
    print_string = f"| power_method: {largest_singular_value:.6f} | numpy svd: {numpy_result:.6f} | iteration:{power_iteration}|"
    print(print_string)