import torch

# Create a tensor to scatter into
x = torch.zeros(5)

# Indices to scatter to
index = torch.tensor([1, 3, 0])

# Values to scatter
src = torch.tensor([2.0, 5.0, 3.0])

# Scatter the values into x at the specified indices
x.scatter_(0, index, src)

print(x)