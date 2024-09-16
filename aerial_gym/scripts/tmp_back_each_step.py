import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的MLP模型
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 设置模型参数
input_size = 10
hidden_size = 20
output_size = 1
seq_length = 5

# 创建模型
model = SimpleMLP(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 输入数据
inputs = torch.randn(seq_length, input_size)
targets = torch.randn(seq_length, output_size)

# 清空梯度图
optimizer.zero_grad()

# 每个时间步进行一次前向传播和反向传播
for t in range(seq_length):
    # 选取当前时间步的输入
    input_t = inputs[t:t+1, :]
    target_t = targets[t:t+1, :]

    # 前向传播
    output_t = model(input_t)

    # 计算损失
    loss = criterion(output_t, target_t)

    # 反向传播
    loss.backward(retain_graph=True)  # 保留计算图

    # 打印当前时间步的梯度
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Step {t}, {name} grad: {param.grad.norm().item()}")

# 更新模型参数
optimizer.step()
