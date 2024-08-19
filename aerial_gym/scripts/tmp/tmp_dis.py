import torch
import torch.nn as nn

# qua = torch.tensor([-0.4380, -0.1426,  0.2417,  0.8541])
batch_size = 2
dim = 3
tensor1 = torch.randn(batch_size, dim)
tensor2 = torch.randn(batch_size, dim)
criterion = nn.MSELoss()
res1 = criterion(tensor1, tensor2)

res2 = torch.sum(torch.sqrt(torch.sum((tensor1 - tensor2) ** 2, dim=1))) / batch_size

res3 = torch.sum(torch.norm(tensor1 - tensor2, dim=1, p=2)) / batch_size
print(res1, res2, res3)

"""
         -7.0653e-03,  5.3359e-01,  4.1898e-01]], device='cuda:0')
tensor([[ 2.2384e+00,  7.1335e+00, -1.5987e+00, -6.4947e-01,  1.0620e+00,
          3.8556e+00, -1.8260e-01,  4.2560e-01, -6.1762e+00, -7.0653e-03,
          5.3359e-01,  4.1898e-01]], device='cuda:0')
    Step 72: loss = 62.338348388671875, distance between sim and dyn = 54.71865463256836
tensor([[ 2.2365,  7.1378, -1.6614, -0.4387, -0.1431,  0.2385,  0.8545, -0.1828,
          0.4259, -6.2741, -0.0221,  0.4530,  0.4810]], device='cuda:0')
tensor([[ 2.2365,  7.1378, -1.6614, -0.6448,  1.0749,  3.8509, -0.1828,  0.4259,
         -6.2741, -0.0221,  0.4530,  0.4810]], device='cuda:0')
    Step 73: loss = 68.89492797851562, distance between sim and dyn = 56.25933837890625
tensor([[ 2.2347,  7.1420, -1.7251, -0.4380, -0.1426,  0.2417,  0.8541, -0.1830,
          0.4261, -6.3720, -0.0353,  0.3724,  0.5421]], device='cuda:0')
tensor([[ 2.2347,  7.1420, -1.7251, -0.6423,  1.0883,  3.8460, -0.1830,  0.4261,
         -6.3720, -0.0353,  0.3724,  0.5421]], device='cuda:0')
    Step 74: loss = 75.5172348022461, distance between sim and dyn = 57.69859313964844
"""