""" 
    为env0生成一个
"""
import os
import random
import time

import gym
import isaacgym  # noqa
from isaacgym import gymutil
from isaacgym.torch_utils import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import sys
from datetime import datetime
import pytz

sys.path.append('/home/cgv841/wzm/FYP/AGAPG')
# print(sys.path)
from aerial_gym.envs import *
from aerial_gym.utils import task_registry, velh_loss, velh_lossVer2, velh_lossVer3, velh_lossVer5
from aerial_gym.models import TrackGroundModelVer7
from aerial_gym.envs import IsaacGymDynamics
# os.path.basename(__file__).rstrip(".py")
def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "track_groundVer7", "help": "The name of the task."},
        {"name": "--experiment_name", "type": str, "default": "test_succ__thr2__600in1000", "help": "Name of the experiment to run or load."},
        {"name": "--headless", "action": "store_true", "default": True, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--num_envs", "type": int, "default": 8, "help": "Number of environments to create. Batch size will be equal to this"},
        {"name": "--seed", "type": int, "default": 75831, "help": "Random seed. Overrides config file if provided."},

        # train setting
        # {"name": "--learning_rate", "type":float, "default": 0.0026,
        #     "help": "the learning rate of the optimizer"},
        # {"name": "--batch_size", "type":int, "default": 128,
        #     "help": "batch size of training. Notice that batch_size should be equal to num_envs"},
        # {"name": "--num_worker", "type":int, "default": 4,
        #     "help": "num worker of dataloader"},
        # {"name": "--num_epoch", "type":int, "default": 600,
        #     "help": "num of epoch"},
        # {"name": "--len_sample", "type":int, "default": 50,
        #     "help": "length of a sample"},
        {"name": "--tmp", "type": bool, "default": True, "help": "Set false to officially save the trainning log"},
        # model setting
        {"name": "--param_load_path_track_simple", "type":str, "default": '/home/cgv841/wzm/FYP/AGAPG/aerial_gym/param_saved/track_groundVer17Ver2.pth',
            "help": "The path to model parameters"},

        # test setting
        {"name": "--visual", "action": "store_true", "default": False, "help": "Whether use isaac gym to visual movement"},
        {"name": "--batch_size", "type":int, "default": 8,  "help": "batch size of training. Notice that batch_size should be equal to num_envs"},
        {"name": "--num_epoch", "type":int, "default": 8, "help": "num of epoch"},
        {"name": "--num_worker", "type":int, "default": 4, "help": "num worker of dataloader"},
        {"name": "--len_sample", "type":int, "default": 1000, "help": "length of a sample"},
        
        {"name": "--num_sample", "type":int, "default": 512, "help": "the total number of sample"},
        {"name": "--threshold", "type":float, "default": 2, "help": "the radius to judge in last 100 step"},
        
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

    run_name = f"Test__{args.experiment_name}__{args.seed}__{get_time()}"
    if args.tmp:
        run_name = 'tmp_' + run_name
    writer = SummaryWriter(f"/home/cgv841/wzm/FYP/AGAPG/runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    
    device = args.sim_device
    print("using device:", device)
    envs, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    dynamic = IsaacGymDynamics()
    
    model = TrackGroundModelVer7().to(device)
    checkpoint = torch.load(args.param_load_path_track_simple, map_location=device)
    model.load_state_dict(checkpoint)
    # torch.manual_seed(args.seed)

    total_sample = 0
    fail_sample = 0
    
    ones = torch.ones(args.batch_size, device=device)
    zeros = torch.zeros(args.batch_size, device=device)
    
    with torch.no_grad():
        while total_sample < args.num_sample:
            
            
            reset_buf = None
            now_quad_state = envs.reset(reset_buf=reset_buf, reset_quad_state=None)
            reset_buf = torch.zeros((args.batch_size,))
            
            tar_ori = torch.zeros((args.batch_size, 3)).to(device)
            
            not_reset_buf = torch.ones(args.batch_size).to(device)
            
            out_buf = torch.zeros(args.batch_size, device=device)

            for step in range(args.len_sample):
                image = envs.get_camera_output()
                action = model(now_quad_state[:, 3:], image)
                
                new_state_dyn = dynamic(now_quad_state, action, envs.cfg.sim.dt)
                new_state_sim, tar_state = envs.step(action)
                tar_pos = tar_state[:, :3].detach()
                loss, loss_direction, loss_speed, loss_ori, loss_h = velh_lossVer5(now_quad_state, tar_pos, 7, tar_ori)
                
                now_quad_state = new_state_dyn
                
                
                if (step + 1) % 50 == 0:
                    
                    now_quad_state = envs.reset(reset_buf=reset_buf, reset_quad_state=now_quad_state).detach()
                envs.reset(reset_buf=reset_buf, reset_quad_state=now_quad_state)
                tar_pos[:, 2] = 7
                if step >= 400:
                    dis = torch.norm(now_quad_state[:, :3] - tar_pos, dim=1, p=2)
                    out_buf = torch.where(dis > args.threshold, ones, out_buf)

                # horizon_dis = torch.norm(now_quad_state[0, :2] - tar_pos[0, :2], dim=0, p=2)
                # vertical_dis = torch.abs(7 - now_quad_state[0, 2])
                # speed = torch.norm(now_quad_state[0, 3:6], dim=1, p=2)
                # writer.add_scalar('Horizon Distance', horizon_dis, step)
                # writer.add_scalar('Vertical Distance', vertical_dis, step)
                # writer.add_scalar('Total Loss', loss[0], step)
                # writer.add_scalar('Direction Loss', loss_direction[0], step)
                # writer.add_scalar('Speed Loss', loss_speed[0], step)
                # writer.add_scalar('Orientation Loss', loss_ori[0], step)
                # writer.add_scalar('Speed', speed, step)
                if (total_sample + 8) % 40 == 0:
                    if (step + 1) % 10 == 0:
                        print(f"    Step {step}: tar_pos = {tar_pos[0]}, now_pos = {now_quad_state[0, :3]}, now_evl = {now_quad_state[0, 6:9]}, action = {action[0]}")
                
                
                
                # if not step % 10:
                #     file_name = f'tmp{step}.png'
                #     envs.save_camera_output(file_name=file_name, file_path='/home/cgv841/wzm/FYP/AGAPG/aerial_gym/scripts/camera_output/frames/')
            out_buf_idx = torch.nonzero(out_buf).squeeze(-1)
            num_out = len(out_buf_idx)
            fail_sample += num_out
            total_sample += args.batch_size
            writer.add_scalar('Number of Out', fail_sample, total_sample)
        
    print("Testing Complete!")
            

        