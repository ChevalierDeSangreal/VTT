import os
import random
import time

import isaacgym  # noqa
from isaacgym import gymutil
from isaacgym.torch_utils import *
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


import pytz
from datetime import datetime
import sys
sys.path.append('/home/wangzimo/VTT/VTT')
# print(sys.path)
from aerial_gym.envs import *
from aerial_gym.utils import task_registry, velh_lossVer5, agile_lossVer1, AgileLoss, agile_lossVer2
from aerial_gym.models import TrackAgileModuleVer0, TrackGroundModelVer6
from aerial_gym.envs import IsaacGymDynamics, NewtonDynamics
# os.path.basename(__file__).rstrip(".py")
from pytorch3d.transforms import euler_angles_to_matrix

"""
Test program for train_trackagileVer3.py
"""


def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "track_agileVer0", "help": "The name of the task."},
        {"name": "--experiment_name", "type": str, "default": "track_agileVer0", "help": "Name of the experiment to run or load."},
        {"name": "--headless", "action": "store_true", "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--num_envs", "type": int, "default": 1024, "help": "Number of environments to create. Batch size will be equal to this"},
        {"name": "--seed", "type": int, "default": 142, "help": "Random seed. Overrides config file if provided."},

        # train setting
        {"name": "--learning_rate", "type":float, "default": 1.6e-6,
            "help": "the learning rate of the optimizer"},
        {"name": "--batch_size", "type":int, "default": 1024,
            "help": "batch size of training. Notice that batch_size should be equal to num_envs"},
        {"name": "--num_worker", "type":int, "default": 4,
            "help": "num worker of dataloader"},
        {"name": "--num_epoch", "type":int, "default": 1520,
            "help": "num of epoch"},
        {"name": "--len_sample", "type":int, "default": 150,
            "help": "length of a sample"},
        {"name": "--tmp", "type": bool, "default": False, "help": "Set false to officially save the trainning log"},
        {"name": "--gamma", "type":int, "default": 0.8,
            "help": "how much will learning rate decrease"},
        {"name": "--slide_size", "type":int, "default": 10,
            "help": "size of GRU input window"},
        {"name": "--step_size", "type":int, "default": 100,
            "help": "learning rate will decrease every step_size steps"},

        # model setting
        {"name": "--param_save_path", "type":str, "default": '/home/wangzimo/VTT/VTT/aerial_gym/param_saved/track_agileVer0.pth',
            "help": "The path to model parameters"},
        {"name": "--param_load_path", "type":str, "default": '/home/wangzimo/VTT/VTT/aerial_gym/param_saved/track_agileVer0.pth',
            "help": "The path to model parameters"},
        
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
    run_name = f"{args.task}__{args.experiment_name}__{args.seed}__{get_time()}"
    # print(args.tmp)
    
    if args.tmp:
        run_name = 'tmp_' + run_name
    writer = SummaryWriter(f"/home/wangzimo/VTT/VTT/aerial_gym/test_runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )


    device = args.sim_device
    print("using device:", device)
    # print("Here I am!!!")
    envs, env_cfg = task_registry.make_env(name=args.task, args=args)
    # print("Here I am!!!")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dynamic = IsaacGymDynamics()
    
    # model = TrackAgileModuleVer0(device=device).to(device)
    model = TrackGroundModelVer6(device=device).to(device)
    checkpoint = torch.load(args.param_load_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    criterion = nn.MSELoss(reduction='none')
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    tar_ori = torch.zeros((args.batch_size, 3)).to(device)

    init_vec = torch.tensor([[1.0, 0.0, 0.0]] * args.batch_size, device=device).unsqueeze(-1)

    input_buffer = torch.zeros(args.slide_size, args.batch_size, 9+3).to(device)

    with torch.no_grad():
        for epoch in range(args.num_epoch):
            
            print(f"Epoch {epoch} begin...")
            old_loss = AgileLoss(args.batch_size, device=device)
            timer = torch.zeros((args.batch_size,), device=device)
            reset_buf = None
            now_quad_state = envs.reset(reset_buf=reset_buf).detach()
            if torch.isnan(now_quad_state[:, 3:6]).any():
                # print(input_buffer[max(step+1-args.slide_size, 0):step+1])
                print("Nan detected in early input!!!")
                exit(0)
            reset_buf = torch.zeros((args.batch_size,))
            
            num_reset = 0
            tar_state = now_quad_state.clone().detach()

            # train
            for step in range(args.len_sample):

                # rel_dis = envs.get_relative_distance()
                # tar_state = envs.get_tar_state().detach()
                rel_dis = tar_state[:, :3] - now_quad_state[:, :3]
                
                # real_rel_dis = envs.get_future_relative_distance()

                # action = model(now_quad_state[:, 3:], rel_dis, real_rel_dis)
                tmp_input = torch.cat((now_quad_state[:, 3:], rel_dis), dim=1)
                tmp_input = tmp_input.unsqueeze(0)
                input_buffer = input_buffer[1:].detach()
                input_buffer = torch.cat((input_buffer, tmp_input), dim=0)
                if torch.isnan(now_quad_state[:, 3:]).any():
                    # print(input_buffer[max(step+1-args.slide_size, 0):step+1])
                    print("Nan detected in input!!!")
                    exit(0)
                # action = model(input_buffer.clone().detach())
                action = model(now_quad_state[:, 3:], rel_dis)
                # print(action.shape)
                if torch.isnan(action).any():
                    print("Nan detected in action!!!")
                    exit(0)
                
                # print("Label:0")
                new_state_dyn, acceleration = dynamic(now_quad_state, action, envs.cfg.sim.dt)
                # print("Label:0.25")
                new_state_sim, tmp_tar_state = envs.step(new_state_dyn.detach())
                # print("Label:0.5")
                tar_pos = tar_state[:, :3].detach()
                
                now_quad_state = new_state_dyn

                # print("Label:1")
                reset_buf, reset_idx = envs.check_reset_out()
                # if len(reset_idx):
                #     print(f"On step {step}, reset {reset_idx}")
                not_reset_buf = torch.logical_not(reset_buf)
                num_reset += len(reset_idx)
                input_buffer[:, reset_idx] = 0

                # loss, loss_direction, loss_speed, loss_h, loss_ori, loss_intent = space_lossVer2(now_quad_state, tar_state, predict_rel_dis, real_rel_dis, tar_pos, 7, tar_ori, criterion)
                # loss, loss_direction, loss_speed, loss_h, loss_ori = space_lossVer4(now_quad_state, tar_state, tar_pos, 7, tar_ori)
                # loss, loss_direction, loss_speed, loss_ori, loss_h = velh_lossVer5(now_quad_state, tar_pos, 7, tar_ori)
                # loss, loss_direction, loss_distance, loss_velocity, loss_ori, loss_h = agile_lossVer1(now_quad_state, tar_state, 7, tar_ori, 1, step, envs.cfg.sim.dt, init_vec)
                loss, new_loss = agile_lossVer2(old_loss, now_quad_state, tar_state, 7, tar_ori, 1, timer, envs.cfg.sim.dt, init_vec)
                old_loss = new_loss
                # print("Label:2")
                
            
                now_quad_state[reset_idx] = envs.reset(reset_buf=reset_buf)[reset_idx].detach()
                


                ave_loss_direciton = torch.sum(old_loss.direction) / args.batch_size
                ave_loss_distance = torch.sum(old_loss.distance) / args.batch_size
                ave_loss_velocity = torch.sum(old_loss.vel) / args.batch_size
                ave_loss_ori = torch.sum(old_loss.ori) / args.batch_size
                ave_loss_h = torch.sum(old_loss.h) / args.batch_size
                # ave_loss_intent = torch.sum(loss_intent) / args.batch_size
                ave_loss = torch.sum(loss) / args.batch_size

                rotation_matrices = euler_angles_to_matrix(now_quad_state[:, 3:6], convention='XYZ')
                direction_vector = rotation_matrices @ init_vec
                direction_vector = direction_vector.squeeze()

                cos_sim = F.cosine_similarity(direction_vector, rel_dis, dim=1)
                theta = torch.acos(cos_sim)
                theta_degrees = theta * 180.0 / torch.pi

                horizon_dis = torch.norm(now_quad_state[0, :2] - tar_pos[0, :2], dim=0, p=2)
                speed = torch.norm(now_quad_state[0, 6:9], dim=0, p=2)
                if reset_buf[0]:
                    loss[0] = float('nan')
                writer.add_scalar(f'Total Loss', loss[0], step)
                writer.add_scalar(f'Direction Loss', old_loss.direction[0], step)
                writer.add_scalar(f'Distance Loss', old_loss.distance[0], step)
                writer.add_scalar(f'Velocity Loss', old_loss.vel[0], step)
                writer.add_scalar(f'Orientation Loss', old_loss.ori[0], step)
                writer.add_scalar(f'Height Loss', old_loss.h[0], step)

                writer.add_scalar(f'Orientation/X', direction_vector[0, 0], step)
                writer.add_scalar(f'Orientation/Y', direction_vector[0, 1], step)
                writer.add_scalar(f'Orientation/Z', direction_vector[0, 2], step)
                writer.add_scalar(f'Orientation/Theta', theta_degrees[0], step)
                writer.add_scalar(f'Acceleration/X', acceleration[0, 0], step)
                writer.add_scalar(f'Acceleration/Y', acceleration[0, 1], step)
                writer.add_scalar(f'Acceleration/Z', acceleration[0, 2], step)
                writer.add_scalar(f'Horizon Distance', horizon_dis, step)
                writer.add_scalar(f'Target Position/X', tar_pos[0, 0], step)
                writer.add_scalar(f'Target Position/Y', tar_pos[0, 1], step)
                writer.add_scalar(f'Position/X', now_quad_state[0, 0], step)
                writer.add_scalar(f'Position/Y', now_quad_state[0, 1], step)
                writer.add_scalar(f'Velocity/X', now_quad_state[0, 6], step)
                writer.add_scalar(f'Velocity/Y', now_quad_state[0, 7], step)
                writer.add_scalar(f'Distance/X', tar_pos[0, 0] - now_quad_state[0, 0], step)
                writer.add_scalar(f'Distance/Y', tar_pos[0, 1] - now_quad_state[0, 1], step)
                writer.add_scalar(f'Action/F', action[0, 0], step)
                writer.add_scalar(f'Action/X', action[0, 1], step)
                writer.add_scalar(f'Action/Y', action[0, 2], step)
                writer.add_scalar(f'Action/Z', action[0, 3], step)
                writer.add_scalar(f'Speed/Z', now_quad_state[0, 8], step)
                writer.add_scalar(f'Speed', speed, step)
                writer.add_scalar(f'Height', now_quad_state[0, 2], step)

                old_loss.reset(reset_idx=reset_idx)
                timer = timer + 1
                timer[reset_idx] = 0
            print(f"Epoch {epoch}, Ave loss = {ave_loss}, num reset = {num_reset}")
            break
            
    
    writer.close()
    print("Testing Complete!")
