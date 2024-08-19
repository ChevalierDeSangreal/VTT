import torch

tensor1 = torch.tensor([1, 1, 1])
tensor2 = torch.tensor([2, 2, 2])
tensor3 = tensor1[:]
tensor3.copy_(tensor2)
tensor3[0] = 3
print(tensor1)