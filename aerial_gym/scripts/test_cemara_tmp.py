import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random

import isaacgym  # noqa
from isaacgym import gymutil
from isaacgym.torch_utils import *
import numpy as np
import torch
import sys



sys.path.append('/home/cgv841/wzm/FYP/AGAPG')
# print(sys.path)
from aerial_gym.envs import *
from aerial_gym.utils import task_registry
from aerial_gym.models import TrackGroundModelVer2

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "track_ground_test", "help": "The name of the task."},
        {"name": "--experiment_name", "type": str, "default": os.path.basename(__file__).rstrip(".py"), "help": "Name of the experiment to run or load."},
        {"name": "--headless", "action": "store_true", "default": True, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create. Batch size will be equal to this"},
        {"name": "--seed", "type": int, "default": 42, "help": "Random seed. Overrides config file if provided."},

        # train setting
        {"name": "--learning_rate", "type":float, "default": 0.1,
            "help": "the learning rate of the optimizer"},
        {"name": "--batch_size", "type":int, "default": 1,
            "help": "batch size of training. Notice that batch_size should be equal to num_envs"},
        {"name": "--num_worker", "type":int, "default": 4,
            "help": "num worker of dataloader"},
        {"name": "--num_epoch", "type":int, "default": 1000,
            "help": "num of epoch"},
        {"name": "--len_sample", "type":int, "default": 100,
            "help": "length of a sample"},

        {"name": "--param_path_dynamic", "type":str, "default": '/home/cgv841/wzm/FYP/AGAPG/aerial_gym/param_saved/dynamic_learntVer2.pth',
            "help": "The path to dynamic model parameters"},
        ]

    # parse arguments
    args = gymutil.parse_arguments(
        description="APG Policy",
        custom_parameters=custom_parameters)


    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"


    return args



if __name__ == "__main__":
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.sim_device
    print("using device:", device)
    model = TrackGroundModelVer2(device=device).to(device)
    print(f"device = {device}")
    dynamic = LearntDynamics(device=device, param_file_path=args.param_path_dynamic)
    dynamic.load_parameters()
    dynamic.to(device)

    envs, env_cfg = task_registry.make_env(name=args.task, args=args)


    # dataset = QuadGroundDataset(args.num_sample, args.len_sample, device)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    now_state = envs.reset()
    
    action = torch_rand_float(-1.0, 1.0, (args.batch_size, 4), device)
    np.set_printoptions(threshold=np.inf)
    for step in range(args.len_sample):
        
        
        
            
        # loss = criterion(new_state_dyn[:, :3], tar_pos)

        # optimizer.step()
        new_state_sim = envs.step(action)
        # 
        if step == 0:
            tmp = envs.get_camera_dep_output()
            print(torch.sum(tmp, dim=(1, 2)))
            x = envs.save_camera_output()
            # print(x)
            is_all_zero = np.all(x == 0)
            print(is_all_zero)
    envs.reset()
    print("Testing Complete!")
            

        