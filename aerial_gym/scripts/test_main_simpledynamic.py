""" 
    为env0生成一个
"""
import os
import random
import time

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

sys.path.append('/home/wangzimo/VTT/VTT')
# print(sys.path)
from aerial_gym.envs import *
from aerial_gym.utils import task_registry, space_lossVer5, Loss, velh_lossVer5
from aerial_gym.models import TrackSpaceModuleVer4
from aerial_gym.envs import SimpleDynamics
# os.path.basename(__file__).rstrip(".py")
def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "track_spaceVer2", "help": "The name of the task."},
        {"name": "--experiment_name", "type": str, "default": "test_height__noreduction", "help": "Name of the experiment to run or load."},
        {"name": "--headless", "action": "store_true", "default": True, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--num_envs", "type": int, "default": 1024, "help": "Number of environments to create. Batch size will be equal to this"},
        {"name": "--seed", "type": int, "default": 7251, "help": "Random seed. Overrides config file if provided."},

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
        {"name": "--param_load_path", "type":str, "default": '/home/wangzimo/VTT/VTT/aerial_gym/param_saved/track_spaceVer0_simple_reduction_100h.pth',
            "help": "The path to model parameters"},

        # test setting
        {"name": "--visual", "action": "store_true", "default": False, "help": "Whether use isaac gym to visual movement"},
        {"name": "--batch_size", "type":int, "default": 1024,  "help": "batch size of training. Notice that batch_size should be equal to num_envs"},
        {"name": "--num_epoch", "type":int, "default": 8, "help": "num of epoch"},
        {"name": "--num_worker", "type":int, "default": 16, "help": "num worker of dataloader"},
        {"name": "--len_sample", "type":int, "default": 150, "help": "length of a sample"},
        
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
    writer = SummaryWriter(f"/home/wangzimo/VTT/VTT/aerial_gym/test_runs/{run_name}")
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
    
    dynamic = SimpleDynamics()
    
    model = TrackSpaceModuleVer4(device=device).to(device)
    checkpoint = torch.load(args.param_load_path, map_location=device)
    model.load_state_dict(checkpoint)

    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    criterion = nn.MSELoss(reduction='none')
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    tar_ori = torch.zeros((args.batch_size, 3)).to(device)

    with torch.no_grad():
        for epoch in range(args.num_epoch):
            print(f"Epoch {epoch} begin...")
            
            reset_buf = None
            now_quad_state = envs.reset(reset_buf=reset_buf).detach()

            reset_buf = torch.zeros((args.batch_size,))
            reset_idx = []
            timer = torch.zeros((args.batch_size,), device=device)
            
            num_reset = 0
            
            sum_loss = 0
            num_loss = 0

            loss = Loss(args.batch_size, device)

            # train
            for step in range(args.len_sample):
                # rel_dis = envs.get_relative_distance()
                # print("##### 0")
                tar_state = envs.get_tar_state().detach()
                rel_dis = now_quad_state[:, :3] - tar_state[:, :3]
                
                # real_rel_dis = envs.get_future_relative_distance()

                # action = model(now_quad_state[:, 3:], rel_dis, real_rel_dis)
                # print("##### 1")
                action = model(now_quad_state[:, 6:9], rel_dis)
                if torch.isnan(action).any():
                    print("Nan detected!!!")
                    exit(0)
                
                # if step == 25 or step == 26 or step == 27:
                #     print(action[0])
                # if step == 53 or step == 54 or step == 55:
                #     print(action[0])
                new_state_dyn, acceleration = dynamic(now_quad_state, action, envs.cfg.sim.dt)
                # if step == 25 or step == 26 or step == 27:
                #     print(acceleration[0])
                # if step == 53 or step == 54 or step == 55:
                #     print(acceleration[0])
                # print("##### 2")
                if not step:
                    last_acceleration = acceleration
                else:
                    last_acceleration[reset_idx] = acceleration[reset_idx]
                # new_state_dyn = torch.cat((action, action.clone().detach(), action.clone().detach()), dim=1)

                new_state_sim, tar_state = envs.step(new_state_dyn.detach())
                # print("##### 6")
                tar_pos = tar_state[:, :3].detach()
                
                
                now_quad_state = new_state_dyn
                
                # if (epoch + 1) % 5 == 0:
                #     tar_pos[:, 2] = 7
                #     if step > args.len_sample - 100:
                #         scaled_now_quad_pos = torch.max(new_state_dyn, torch.tensor(-10, device=device))
                #         scaled_now_quad_pos = torch.min(scaled_now_quad_pos, torch.tensor(10, device=device))
                #         dis = torch.sum(torch.norm(tar_pos - now_quad_state[:, :3], p=2, dim=1)) / args.batch_size

                # if (step + 1) % 50 == 0:
                reset_buf, reset_idx = envs.check_reset_out()
                # if len(reset_idx):
                #     print(f"On step {step}, reset {reset_idx}")
                not_reset_buf = torch.logical_not(reset_buf)
                num_reset += len(reset_idx)

                    
                # print("##### 9")
                loss_final, loss = space_lossVer5(loss, now_quad_state, acceleration, last_acceleration, tar_state, 7, tar_ori, timer, envs.cfg.sim.dt)
                last_acceleration = acceleration
                # print(f"loss_final shape:{loss_final.shape}")
                # loss_final.backward(not_reset_buf, retain_graph=True)
                # print("##### 10")
                sum_loss += torch.sum(torch.mul(not_reset_buf, loss_final))
                num_loss += args.batch_size - len(reset_idx)

                    
                # print("##### 11")
                now_quad_state[reset_idx] = envs.reset(reset_buf=reset_buf)[reset_idx].detach()
                

                horizon_dis = torch.norm(now_quad_state[0, :2] - tar_pos[0, :2], dim=0, p=2)
                speed = torch.norm(now_quad_state[0, 6:9], dim=0, p=2)
                
                if reset_buf[0]:
                    loss_final[0] = float('nan')
                writer.add_scalar(f'Total Loss', loss_final[0], step)
                writer.add_scalar(f'Direction Loss', loss.direction[0], step)
                writer.add_scalar(f'Speed Loss', loss.speed[0], step)
                writer.add_scalar(f'Orientation Loss', loss.ori[0], step)
                writer.add_scalar('Acceleration Loss', loss.acc[0], step)
                writer.add_scalar('Jerk Loss', loss.jerk[0], step)
                
                writer.add_scalar(f'Acceleration/X', acceleration[0, 0], step)
                writer.add_scalar(f'Acceleration/Y', acceleration[0, 1], step)
                writer.add_scalar(f'Acceleration/Z', acceleration[0, 2], step)
                writer.add_scalar(f'Horizon Distance', horizon_dis, step)
                writer.add_scalar(f'Position/X', now_quad_state[0, 0], step)
                writer.add_scalar(f'Position/Y', now_quad_state[0, 1], step)
                writer.add_scalar(f'Distance/X', tar_pos[0, 0] - now_quad_state[0, 0], step)
                writer.add_scalar(f'Distance/Y', tar_pos[0, 1] - now_quad_state[0, 1], step)
                writer.add_scalar(f'Action/X', action[0, 0], step)
                writer.add_scalar(f'Action/Y', action[0, 1], step)
                writer.add_scalar(f'Action/Z', action[0, 2], step)
                writer.add_scalar(f'Speed/Z', now_quad_state[0, 8], step)
                writer.add_scalar(f'Speed', speed, step)
                writer.add_scalar(f'Height', now_quad_state[0, 2], step)
                loss.reset(reset_idx=reset_idx)
                timer = timer + 1
                timer[reset_idx] = 0
            # print("##### 15")
            # optimizer.step()
            # print("##### 16")
            # optimizer.zero_grad()


            
            

            # print("##### 17")

            
            # envs.update_target_traj()
            break
    writer.close()
    print("Testing Complete!")
            

        