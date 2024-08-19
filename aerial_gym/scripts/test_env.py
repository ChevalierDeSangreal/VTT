import os
import random
import time

import isaacgym  # noqa
from isaacgym import gymutil
from isaacgym.torch_utils import *
import numpy as np
import torch
import pytz
from datetime import datetime

import sys
sys.path.append('/home/cgv841/wzm/FYP/AGAPG')
# print(sys.path)
from aerial_gym.envs import *
from aerial_gym.utils import task_registry, velh_loss, velh_lossVer2
from aerial_gym.envs import IsaacGymDynamics
# os.path.basename(__file__).rstrip(".py")
def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "track_groundVer7", "help": "The name of the task."},
        {"name": "--experiment_name", "type": str, "default": "exp7", "help": "Name of the experiment to run or load."},
        {"name": "--headless", "action": "store_true", "default": True, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--num_envs", "type": int, "default": 2, "help": "Number of environments to create. Batch size will be equal to this"},
        {"name": "--seed", "type": int, "default": 52, "help": "Random seed. Overrides config file if provided."},

        # train setting
        {"name": "--learning_rate", "type":float, "default": 2.6e-6,
            "help": "the learning rate of the optimizer"},
        {"name": "--batch_size", "type":int, "default": 2,
            "help": "batch size of training. Notice that batch_size should be equal to num_envs"},
        {"name": "--num_worker", "type":int, "default": 4,
            "help": "num worker of dataloader"},
        {"name": "--num_epoch", "type":int, "default": 4000,
            "help": "num of epoch"},
        {"name": "--len_sample", "type":int, "default": 150,
            "help": "length of a sample"},
        {"name": "--tmp", "type": bool, "default": True, "help": "Set false to officially save the trainning log"},
        {"name": "--gamma", "type":int, "default": 0.5,
            "help": "how much will learning rate decrease"},
        {"name": "--step_size", "type":int, "default": 250,
            "help": "learning rate will decrease every step_size steps"},
        ]

    # parse arguments
    args = gymutil.parse_arguments(
        description="APG Policy",
        custom_parameters=custom_parameters)

    assert args.batch_size == args.num_envs, "batch_size should be equal to num_envs"

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"


    return args

def get_time():

    timestamp = time.time()  # 替换为您的时间戳

    # 将时间戳转换为datetime对象
    dt_object_utc = datetime.utcfromtimestamp(timestamp)

    # 指定目标时区（例如"Asia/Shanghai"）
    target_timezone = pytz.timezone("Asia/Shanghai")
    dt_object_local = dt_object_utc.replace(tzinfo=pytz.utc).astimezone(target_timezone)

    # 将datetime对象格式化为字符串
    formatted_time_local = dt_object_local.strftime("%Y-%m-%d %H:%M:%S %Z")

    return formatted_time_local

if __name__ == "__main__":
    args = get_args()

    device = args.sim_device
    print("using device:", device)
    envs, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dynamic = IsaacGymDynamics()
    
    # checkpoint = torch.load(args.param_load_path_track_simple, map_location=device)
    # model.load_state_dict(checkpoint)
    # model = TrackSimplerModel(device=device).to(device)



    now_quad_state = envs.reset(reset_buf=None, reset_quad_state=None).detach()
    action = torch_rand_float(-1.0, 1.0, (args.batch_size, 4), device)
    # action = torch.ones_like(action)
    # action[:, 0] = -1
    # train
    for step in range(args.len_sample):
        
        
        print(f"Step: {step}")
        print("Actions:", action[0])
        new_state_dyn = dynamic(now_quad_state, action, envs.cfg.sim.dt)

        new_state_sim, tar_state = envs.step(action)
        # print(f"In epoch {epoch}, step{step}---------------------------")
        # print("Origin State:", now_quad_state[0])
        # print("Sim State:", new_state_sim[0])
        # print("Dyn State:", new_state_dyn[0])
        tar_pos = tar_state[:, :3].detach()
        
        # print("", new_state_dyn[0])
        print("Dynamic State Position\n", new_state_dyn[0, :3])
        print("Simulation State Position\n", new_state_sim[0, :3])
        print("Dynamic State Orientation\n", new_state_dyn[0, 3:6])
        print("Simulation State Orientation\n", new_state_sim[0, 3:6])
        print("Dynamic State Linear Vel\n", new_state_dyn[0, 6:9])
        print("Simulation State Linear Vel\n", new_state_sim[0, 6:9])
        print("Dynamic State Angular Vel\n", new_state_dyn[0, 9:12])
        print("Simulation State Angular Vel\n", new_state_sim[0, 9:12])
        now_quad_state = new_state_dyn
        # break
            
