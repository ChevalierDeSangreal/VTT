"""
Generate trajectory which consists of segments.
"""
import os
import numpy as np
import torch

class TrajectoryGenerator:
    def __init__(self, speed, sampling_frequency, direction_change_interval, total_time, batch_size, device='cpu'):
        """
        初始化轨迹生成器。
        
        :param speed: 物体的运动速度（标量）。
        :param sampling_frequency: 采样频率 单位是秒。
        :param direction_change_interval: 物体每隔多少秒改变方向。
        :param batch_size: 生成的轨迹数量。
        :param device: 目标设备 ('cpu' 或 'cuda')。
        """
        self.speed = speed
        self.dt = sampling_frequency
        self.direction_change_interval = direction_change_interval
        self.device = device
        self.batch_size = batch_size
        self.total_time = total_time

    def generate_directions(self, num_changes):
        """批量生成方向向量"""
        angles = np.random.uniform(0, 2 * np.pi, size=(self.batch_size, num_changes))
        directions = np.stack((np.cos(angles), np.sin(angles)), axis=-1)  # (batch_size, num_changes, 2)
        return torch.tensor(directions, dtype=torch.float32, device=self.device)

    def batch_generate_trajectories(self):
        """
        批量生成多个轨迹，并将它们转换为目标设备上的张量。
        
        :return: 包含多个轨迹的张量 (batch_size, num_samples, 2)。
        """
        num_samples = int(self.total_time / self.dt)
        change_interval_samples = int(self.direction_change_interval / self.dt)
        num_direction_changes = (num_samples + change_interval_samples - 1) // change_interval_samples
        
        # 生成初始方向和每次变向时的新方向
        directions = self.generate_directions(num_direction_changes)

        # 初始化位置为原点
        positions = torch.zeros((self.batch_size, num_samples, 2), dtype=torch.float32, device=self.device)

        # 当前方向的初始值
        current_direction = directions[:, 0, :]  # (batch_size, 2)

        for i in range(num_samples):
            if i % change_interval_samples == 0 and i // change_interval_samples < num_direction_changes:
                # 更新方向
                current_direction = directions[:, i // change_interval_samples, :]
            
            # 更新位置
            displacement = current_direction * self.speed * self.dt  # (batch_size, 2)
            if i > 0:
                positions[:, i, :] = positions[:, i-1, :] + displacement
            else:
                positions[:, i, :] = displacement
        
        return positions

# # 参数设定
# v = 1.0  # 速度
# dt = 0.1  # 采样频率
# direction_change_interval = 2.0  # 变向间隔
# total_time = 10.0  # 每条轨迹的总时间
# batch_size = 5  # 批量生成的轨迹数量
# device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 目标设备

# # 生成器初始化
# generator = TrajectoryGenerator(v, dt, direction_change_interval, device)

# # 批量生成轨迹并转换为张量
# batch_tensor = generator.batch_generate_trajectories(total_time, batch_size)

# # 输出张量形状
# print(f"生成的轨迹张量形状: {batch_tensor.shape}")
