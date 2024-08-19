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
import sys
import pytz
from datetime import datetime

from torch.optim import lr_scheduler

sys.path.append('/home/cgv841/wzm/FYP/AGAPG')
# print(sys.path)
from aerial_gym.envs import *
from aerial_gym.utils import task_registry
from aerial_gym.models import TrackGroundModelVer7
from aerial_gym.envs import IsaacGymDynamics
# os.path.basename(__file__).rstrip(".py")
def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "track_groundVer7", "help": "The name of the task."},
        {"name": "--experiment_name", "type": str, "default": "exp7__25back", "help": "Name of the experiment to run or load."},
        {"name": "--headless", "action": "store_true", "default": True, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--num_envs", "type": int, "default": 8, "help": "Number of environments to create. Batch size will be equal to this"},
        {"name": "--seed", "type": int, "default": 42, "help": "Random seed. Overrides config file if provided."},

        # train setting
        {"name": "--learning_rate", "type":float, "default": 2.6e-4,
            "help": "the learning rate of the optimizer"},
        {"name": "--batch_size", "type":int, "default": 8,
            "help": "batch size of training. Notice that batch_size should be equal to num_envs"},
        {"name": "--num_worker", "type":int, "default": 4,
            "help": "num worker of dataloader"},
        {"name": "--num_epoch", "type":int, "default": 4000,
            "help": "num of epoch"},
        {"name": "--len_sample", "type":int, "default": 200,
            "help": "length of a sample"},
        {"name": "--tmp", "type": bool, "default": True, "help": "Set false to officially save the trainning log"},
        {"name": "--gamma", "type":int, "default": 0.5,
            "help": "how much will learning rate decrease"},
        {"name": "--step_size", "type":int, "default": 100,
            "help": "learning rate will decrease every step_size steps"},

        # model setting
        {"name": "--param_save_path_track_simple", "type":str, "default": '/home/cgv841/wzm/FYP/AGAPG/aerial_gym/param_saved/track_groundVer18.pth',
            "help": "The path to model parameters"},
        {"name": "--param_load_path_track_simple", "type":str, "default": '/home/cgv841/wzm/FYP/AGAPG/aerial_gym/param_saved/track_groundVer9__len_sample_50.pth',
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
    
    model = TrackGroundModelVer7(device=device).to(device)
    # checkpoint = torch.load(args.param_load_path_track_simple, map_location=device)
    # model.load_state_dict(checkpoint)
    # model = TrackSimplerModel(device=device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    criterion = nn.MSELoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)


    for epoch in range(args.num_epoch):
        print(f"Epoch {epoch} begin...")
        optimizer.zero_grad()
        
        reset_buf = None
        now_quad_state = envs.reset(reset_buf=reset_buf, reset_quad_state=None).detach()
        reset_buf = torch.zeros((args.batch_size,))
                    
        # train
        for step in range(args.len_sample):
            
            image = envs.get_camera_output()
            
            action = model(now_quad_state[:, 3:], image)
            
            # action = torch_rand_float(-1.0, 1.0, (args.batch_size, 4), device)
            # print(action, now_state)
            new_state_dyn = dynamic(now_quad_state, action, envs.cfg.sim.dt)
            
            new_state_sim, tar_state = envs.step(action)
            tar_pos = tar_state[:, :3]
            
            now_quad_state = new_state_dyn

            if (step + 1) % 50 == 0:
                # print("Here I am!!")
                reset_buf, reset_idx = envs.check_reset_out()
                if len(reset_idx):
                    print(f"On step {step}, reset {reset_idx}")
                not_reset_buf = torch.logical_not(reset_buf)
                
                loss1 = torch.norm(now_quad_state[:, :2] - tar_pos[:, :2], dim=1, p=2)
                loss2 = torch.abs(now_quad_state[:, 2] - 7)
                loss = loss1 + 0.4 * loss2
                
                if args.batch_size - len(reset_idx):
                    loss = torch.sum(loss) / args.batch_size
                    loss.backward()
                else:
                    loss.backward(not_reset_buf)
                # for name, param in model.hidden_layer3.named_parameters():
                #     writer.add_histogram('hidden_layer3_' + name + '_param', param, epoch)  # 记录参数
                #     if param.grad is not None:
                #         writer.add_histogram('hidden_layer5_' + name + '_grad', param.grad, epoch)  # 记录梯度
                # for name, param in model.hidden_layer5.named_parameters():
                #     writer.add_histogram('hidden_layer5_' + name + '_param', param, epoch)  # 记录参数
                #     if param.grad is not None:
                #         writer.add_histogram('hidden_layer5_' + name + '_grad', param.grad, epoch)  # 记录梯度

                # max_norm = 1.0  # 设置梯度裁剪的阈值
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                optimizer.zero_grad()
                now_quad_state = now_quad_state.detach()
                
            if (epoch + 1) % 10 == 0:
                if (step + 1) % 10 == 0:
                    print(f"    Step {step}: average loss = {ave_loss}, tar_pos = {tar_pos[0]}, now_pos = {now_quad_state[0, :3]}, now_evl = {now_quad_state[0, 6:9]}, action = {action[0]}")
                    
                
            # if (step + 1) % 25 == 0:
            #     if (step + 1) % 50 == 0:
            #         now_quad_state = envs.reset(reset_buf=reset_buf, reset_quad_state=now_quad_state).detach()
            #         reset_buf = torch.zeros((args.batch_size,))
            #     else:
            #         assert torch.sum(reset_buf) == 0, "Reset buf should be all zero but not"
            #         envs.reset(reset_buf=reset_buf, reset_quad_state=now_quad_state)

            if (step + 1) % 50 == 0:
                now_quad_state = envs.reset(reset_buf=reset_buf, reset_quad_state=now_quad_state).detach()
                reset_buf = torch.zeros((args.batch_size,))
                    
                
            

            

            
        # dis_sim_dyn = torch.norm(new_state_dyn[:, :2] - new_state_sim[:, :2], p=2, dim=1)    
        # print("Distance between sim and dynamics:", dis_sim_dyn)
        # exit(0)
        
        # scaled_now_quad_pos = torch.max(new_state_dyn, torch.tensor(-10, device=device))
        # scaled_now_quad_pos = torch.min(scaled_now_quad_pos, torch.tensor(10, device=device))
        # loss1 = criterion(scaled_now_quad_pos[:, :2], tar_pos[:, :2])
        # loss2 = torch.sum(torch.abs(scaled_now_quad_pos[:, 2] - 5)) / args.batch_size
        # loss = 0.5 * loss1 + loss2
        scheduler.step()
        
        # validation, no reset_out
        
        if (epoch + 1) % 5 == 0:
            model.eval()
            
            with torch.no_grad():
                envs.step(action)
                
                reset_buf = None
                now_quad_state = envs.reset(reset_buf=reset_buf, reset_quad_state=None).detach()
                reset_buf = torch.zeros((args.batch_size,))
                    
                for step in range(args.len_sample):
                    
                    image = envs.get_camera_output()
                    
                    action = model(now_quad_state[:, 3:], image)
                    
                    new_state_dyn = dynamic(now_quad_state, action, envs.cfg.sim.dt)
                    new_state_sim, tar_state = envs.step(action)
                    tar_pos = tar_state[:, :3]
                    
                    now_quad_state = new_state_dyn
                    
                    if (step + 1) % 25 == 0:
                        assert torch.sum(reset_buf) == 0, "Reset buf should be all zero but not"
                        envs.reset(reset_buf=reset_buf, reset_quad_state=now_quad_state)
                    
                scaled_now_quad_pos = torch.max(new_state_dyn, torch.tensor(-10, device=device))
                scaled_now_quad_pos = torch.min(scaled_now_quad_pos, torch.tensor(10, device=device))
                loss1 = torch.norm(scaled_now_quad_pos[:, :2] - tar_pos[:, :2], dim=1, p=2)
                loss2 = torch.abs(scaled_now_quad_pos[:, 2] - 7)
                loss = loss1 + 0.4 * loss2
                ave_loss = torch.sum(loss) / args.batch_size
                dis_hoz = torch.sum(torch.norm(scaled_now_quad_pos[:, :2] - tar_pos[:, :2], dim=1, p=2)) / args.batch_size
                dis_ver = torch.sum(torch.abs(scaled_now_quad_pos[:, 2] - 7)) / args.batch_size
                print(f"Epoch {epoch}: loss = {ave_loss}, ver dis = {dis_ver}, hor dis = {dis_hoz}")
                writer.add_scalar('Loss', ave_loss.item(), epoch)
                writer.add_scalar('Vertical Distance', dis_ver.item(), epoch)
                writer.add_scalar('Horizen Distance', dis_hoz.item(), epoch)
            model.train()
        
            
        if (epoch + 1) % 100 == 0:
            print("Saving Model...")
            torch.save(model.state_dict(), args.param_save_path_track_simple)
            # break
            # dynamic.save_parameters()
    
    writer.close()
    print("Training Complete!")
            

        