import sys
sys.path.append('/home/wangzimo/VTT/VTT')

import isaacgym  # noqa
from isaacgym import gymutil
from isaacgym.torch_utils import *

from aerial_gym.envs import IsaacGymDynamics
import torch
from pytorch3d.transforms import euler_angles_to_matrix

if __name__ == "__main__":
    # torch.set_printoptions(precision=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    action = [[0,  0,  0, 1]]
    action = torch.tensor(action).to(device)

    state = [[
        .0, .0, .0, .0, .0, .0, .0,
        .0, .0, .0, .0, .0
    ]]
    state = torch.tensor(state).to(device)
    # state = [2, 3, 4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    dyn = IsaacGymDynamics()
    num_step = 500
    init_vec = torch.tensor([[1.0, 0.0, 0.0]], device=device).unsqueeze(-1)
    for step in range(num_step):
        state, acceleration = dyn.simulate_quadrotor(
            action, state, 0.01
        )
        output_action = dyn.control_quadrotor(action, state)
        # print("action", output_action)

        ori = state[:, 3:6].clone()
        rotation_matrices = euler_angles_to_matrix(ori, convention='XYZ')
        direction_vector = rotation_matrices @ init_vec
        direction_vector = direction_vector.squeeze()
        if not (step % 10): 
            print("======================")
            print(f"Step {step + 1}: new state flightmare", state)
            print(f"Step {step + 1}: direction vec {direction_vector}")