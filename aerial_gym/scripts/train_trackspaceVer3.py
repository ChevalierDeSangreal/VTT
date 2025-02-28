import os
import random
import time

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import isaacgym  # noqa
from isaacgym import gymutil
from isaacgym.torch_utils import *
import numpy as np
import torch
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
from aerial_gym.utils import task_registry, space_lossVer5, Loss, velh_lossVer5
from aerial_gym.models import TrackSpaceModuleVer3, TrackGroundModelVer6
from aerial_gym.envs import SimpleDynamics, NRSimpleDynamics
# os.path.basename(__file__).rstrip(".py")
torch.autograd.set_detect_anomaly(True)

"""
    On the base of Ver2
    Change dynamics to simple dynamics
    Change model to sequential model
"""


def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "track_spaceVer2", "help": "The name of the task."},
        {"name": "--experiment_name", "type": str, "default": "simple_dynamicsVer0_noreduction", "help": "Name of the experiment to run or load."},
        {"name": "--headless", "action": "store_true", "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--num_envs", "type": int, "default": 64, "help": "Number of environments to create. Batch size will be equal to this"},
        {"name": "--seed", "type": int, "default": 42, "help": "Random seed. Overrides config file if provided."},

        # train setting
        {"name": "--learning_rate", "type":float, "default": 5.6e-5,
            "help": "the learning rate of the optimizer"},
        {"name": "--batch_size", "type":int, "default": 64,
            "help": "batch size of training. Notice that batch_size should be equal to num_envs"},
        {"name": "--num_worker", "type":int, "default": 4,
            "help": "num worker of dataloader"},
        {"name": "--num_epoch", "type":int, "default": 1502,
            "help": "num of epoch"},
        {"name": "--len_sample", "type":int, "default": 150,
            "help": "length of a sample"},
        {"name": "--tmp", "type": bool, "default": False, "help": "Set false to officially save the trainning log"},
        {"name": "--gamma", "type":int, "default": 0.8,
            "help": "how much will learning rate decrease"},
        {"name": "--step_size", "type":int, "default": 100,
            "help": "learning rate will decrease every step_size steps"},

        # model setting
        {"name": "--param_save_path", "type":str, "default": '/home/wangzimo/VTT/VTT/aerial_gym/param_saved/track_spaceVer0.pth',
            "help": "The path to model parameters"},
        {"name": "--param_load_path", "type":str, "default": '/home/wangzimo/VTT/VTT/aerial_gym/param_saved/track_spaceVer0_120epoch.pth',
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
    writer = SummaryWriter(f"/home/wangzimo/VTT/VTT/aerial_gym/runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    # args.headless = True

    device = args.sim_device
    print("using device:", device)
    # print("Here I am!!!")
    envs, env_cfg = task_registry.make_env(name=args.task, args=args)
    # print("Here I am!!!")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dynamic = SimpleDynamics()
    
    model = TrackSpaceModuleVer3(device=device).to(device)
    # checkpoint = torch.load(args.param_load_path, map_location=device)
    # model.load_state_dict(checkpoint)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    criterion = nn.MSELoss(reduction='none')
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    tar_ori = torch.zeros((args.batch_size, 3)).to(device)

    for epoch in range(args.num_epoch):
        print(f"Epoch {epoch} begin...")
        optimizer.zero_grad()
        
        reset_buf = None
        now_quad_state = envs.reset(reset_buf=reset_buf).detach()

        reset_buf = torch.zeros((args.batch_size,))
        
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
            action = model(now_quad_state[:, 3:], rel_dis)
            if torch.isnan(action).any():
                print("Nan detected!!!")
                exit(0)
            
            
            new_state_dyn, acceleration = dynamic(now_quad_state, action, envs.cfg.sim.dt)
            # print("##### 2")
            if not step:
                last_acceleration = acceleration
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
            loss_final, loss = space_lossVer5(loss, now_quad_state, acceleration, last_acceleration, tar_state, 7, tar_ori, step, envs.cfg.sim.dt)
            last_acceleration = acceleration
            # print(f"loss_final shape:{loss_final.shape}")
            loss_final.backward(not_reset_buf, retain_graph=True)
            # print("##### 10")
            # ave_loss = torch.sum(torch.mul(not_reset_buf, loss_final)) / (args.batch_size - len(reset_idx))
            sum_loss += torch.sum(torch.mul(not_reset_buf, loss_final))
            num_loss += args.batch_size - len(reset_idx)

                
            # print("##### 11")
            now_quad_state[reset_idx] = envs.reset(reset_buf=reset_buf)[reset_idx].detach()
        # print("##### 15")
        optimizer.step()
        # print("##### 16")
        optimizer.zero_grad()

        ave_loss_direciton = torch.sum(loss.direction.detach()) / args.batch_size
        ave_loss_speed = torch.sum(loss.speed.detach()) / args.batch_size
        ave_loss_ori = torch.sum(loss.ori.detach()) / args.batch_size
        ave_loss_h = torch.sum(loss.h.detach()) / args.batch_size
        # ave_loss_intent = torch.sum(loss_intent) / args.batch_size
        ave_loss_final = torch.sum(loss_final.detach()) / args.batch_size
        
        
        writer.add_scalar('Ave Loss', sum_loss / num_loss, epoch)
        writer.add_scalar('Loss Final', ave_loss_final.item(), epoch)
        writer.add_scalar('Loss Direction', ave_loss_direciton.item(), epoch)
        writer.add_scalar('Loss Speed', ave_loss_speed.item(), epoch)
        # writer.add_scalar('Loss Intent', ave_loss_intent.item(), epoch)
        writer.add_scalar('Loss Orientation', ave_loss_ori.item(), epoch)
        writer.add_scalar('Loss Height', ave_loss_h.item(), epoch)
        writer.add_scalar('Number Reset', num_reset, epoch)
        # print("##### 17")

        print(f"Epoch {epoch}, Ave loss = {sum_loss / num_loss}, num reset = {num_reset}")

        
        if (epoch + 1) % 50 == 0:
            print("Saving Model...")
            torch.save(model.state_dict(), args.param_save_path)
        # print("##### 18")
        # envs.update_target_traj()
    writer.close()
    print("Training Complete!")
