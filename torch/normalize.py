import torch 
import numpy as np

a = torch.rand(3)
w = torch.rand(3, 4)

# --- 
a_norm1  = a.norm(1)
w_norm1 = w.norm(1)

# --- p=2/Frobenius
a_norm2 = a.norm(2)
w_norm2 = w.norm(2)

# --- max norm (maximum value)
a_max = a.norm(p=np.inf)
w_max = w.norm(p=np.inf)

# --- nuclear norm  (largest singular value)
w_nuclear = w.norm(p="nuc")

# --- dimension 
w_norm_dim0 = w.norm(dim=0, p=2)
w_norm_dim1 = w.norm(dim=1, p=2)


print("--- vector ---")
print(a_norm1)
print(a_norm2)
print(a_max, a.max())

print("--- matrix ---")
print(w_norm1)
print(w_norm2)
print(w_norm_dim0)
print(w_norm_dim1)
print(w_max, w.max())
print(w_nuclear)