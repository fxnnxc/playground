import torch 

out_dim, in_dim = 10, 3 

weight = torch.rand(10, 3) 
in_vector = torch.rand(3)
out_vector = torch.rand(10) 

mat_mul_in = torch.matmul(weight, in_vector) 
mat_mul_out = torch.matmul(weight.T, out_vector) 

print(mat_mul_in.size())
print(mat_mul_out.size())

