import torch
a = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])

print(a[:, [3, 2, 1, 0]])