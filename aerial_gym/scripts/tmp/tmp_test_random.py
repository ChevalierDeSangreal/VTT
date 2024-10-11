
import sys

sys.path.append('/home/wangzimo/VTT/VTT')
from aerial_gym.utils import rand_circle_point

import torch

torch.manual_seed(45)
print(rand_circle_point(2, 5, 'cuda:0'))
