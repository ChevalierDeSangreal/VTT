import torch
import torch.nn as nn

# 定义输入
input_tensor = torch.randn(1, 3, 224, 224)

# 多层 pooling
pool = nn.Sequential(
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
)

output = pool(input_tensor)
print(output.shape)  # torch.Size([1, 3, 14, 14])