# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from torch import Tensor

@torch.jit.script
def compute_vee_map(skew_matrix):
    # type: (Tensor) -> Tensor

    # return vee map of skew matrix
    vee_map = torch.stack(
        [-skew_matrix[:, 1, 2], skew_matrix[:, 0, 2], -skew_matrix[:, 0, 1]], dim=1)
    return vee_map

def rand_circle_point(batch_size, r, device):
    thetas = torch.rand(batch_size, device=device) * 2 * torch.tensor(np.pi, device=device)
    x = r * torch.cos(thetas)
    y = r * torch.sin(thetas)
    # print(torch.stack([x, y], dim=1), torch.initial_seed())
    return torch.stack([x, y], dim=1)

def set_circle_point(batch_size, r, device, thetas):
    """
    Generate a position on the circle
    r is the radius of circle
    thetas is the direction
    thetas should be [0, 2pi]
    """
    # thetas = torch.rand(batch_size, device=device) * 2 * torch.tensor(np.pi, device=device)
    thetas = thetas * torch.ones(batch_size, device=device) /180 * torch.tensor(np.pi, device=device)
    x = r * torch.cos(thetas)
    y = r * torch.sin(thetas)
    # print(torch.stack([x, y], dim=1), torch.initial_seed())
    return torch.stack([x, y], dim=1)

# if __name__ == "__main__":
#     print(rand_circle_point(2, 5, 'cuda:0'))