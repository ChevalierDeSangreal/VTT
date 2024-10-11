import sys
sys.path.append('/home/wangzimo/VTT/VTT')
from aerial_gym.models import TrackAgileModuleVer0
import torch
import torch.nn as nn
model = TrackAgileModuleVer0()

model_path = '/home/wangzimo/VTT/VTT/aerial_gym/param_saved/track_agileVer0.pth'
model.load_state_dict(torch.load(model_path))

print("Model's state_dict:")
for name, param in model.state_dict().items():
    print(f"Layer: {name} | Weights: {param}")