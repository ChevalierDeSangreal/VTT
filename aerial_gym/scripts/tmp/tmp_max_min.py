import torch
tensor1 = torch.tensor([[1, 2, 3], [2, 3, 4]])
tensor1 = torch.min(tensor1, torch.tensor(2))
print(tensor1)