from torch.utils.data import Dataset
import os
import numpy as np
import torch

from aerial_gym.scripts.generate_data import load_training_data

class TargetDataset(Dataset):
    def __init__(self, directory, device):
        """
        Args:
            directory (string): Directory with datasets.
        """
        self.directory = directory
        self.data_list = load_training_data(directory)
        self.device = device
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # print("Data Index:", idx)

        data = self.data_list[idx]
        
        target_states = torch.tensor(data, dtype=torch.float32, device=self.device)
        # print("Shape of target states:", target_states.shape)

        return target_states
