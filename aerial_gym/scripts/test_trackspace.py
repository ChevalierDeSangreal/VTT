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
from torch.utils.tensorboard import SummaryWriter
import sys
from datetime import datetime
import pytz

sys.path.append('/home/wangzimo/VTT/VTT')
# print(sys.path)
from aerial_gym.envs import *
from aerial_gym.utils import task_registry, space_lossVer0, velh_lossVer5
from aerial_gym.models import TrackSpaceModuleVer0, TrackSpaceModuleVer2
from aerial_gym.envs import IsaacGymDynamics
# os.path.basename(__file__).rstrip(".py")
def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "track_spaceVer0", "help": "The name of the task."},
        {"name": "--experiment_name", "type": str, "default": "test_height__h", "help": "Name of the experiment to run or load."},
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
        {"name": "--param_load_path_track_simple", "type":str, "default": '/home/wangzimo/VTT/VTT/aerial_gym/param_saved/track_spaceVer0.pth',
            "help": "The path to model parameters"},

        # test setting
        {"name": "--visual", "action": "store_true", "default": False, "help": "Whether use isaac gym to visual movement"},
        {"name": "--batch_size", "type":int, "default": 8,  "help": "batch size of training. Notice that batch_size should be equal to num_envs"},
        {"name": "--num_epoch", "type":int, "default": 8, "help": "num of epoch"},
        {"name": "--num_worker", "type":int, "default": 4, "help": "num worker of dataloader"},
        {"name": "--len_sample", "type":int, "default": 10000, "help": "length of a sample"},
        
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
    writer = SummaryWriter(f"/home/wangzimo/VTT/VTT/aerial_gym/runs/{run_name}")
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
    
    model = TrackSpaceModuleVer2(device=device).to(device)
    checkpoint = torch.load(args.param_load_path_track_simple, map_location=device)
    model.load_state_dict(checkpoint)
    # torch.manual_seed(args.seed)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, eps=1e-5)
    criterion = nn.MSELoss(reduction='none')

    total_sample = 0
    fail_sample = 0

    tar_ori = torch.zeros((args.batch_size, 3)).to(device)
    
    with torch.no_grad():
        for epoch in range(args.num_epoch):
            print(f"Epoch {epoch} begin...")
            
            reset_buf = None
            now_quad_state = envs.reset(reset_buf=reset_buf, reset_quad_state=None)
            reset_buf = torch.zeros((args.batch_size,))
            
            num_reset = 0
            
            sum_loss = 0
            num_loss = 0
            

            for step in range(args.len_sample):

                # rel_dis = envs.get_relative_distance()
                # real_rel_dis = envs.get_future_relative_distance()
                tar_state = envs.get_tar_state()
                rel_dis = now_quad_state[:, :3] - tar_state[:, :3]
                action = model(now_quad_state[:, 3:], rel_dis)

                new_state_dyn = dynamic(now_quad_state, action, envs.cfg.sim.dt)

                new_state_sim, tar_state = envs.step(action)

                tar_pos = tar_state[:, :3].detach()
                
                now_quad_state = new_state_dyn
                # if (step) % 50 == 0:
                #     print(f"Step{step}, After: ~~~~~~~~~~~~~~~~")
                #     print("State of dynamic: ", new_state_dyn[0])
                #     print("State of simulator", new_state_sim[0]) 
                
                if (step + 1) % 50 == 0:
  
                    reset_buf, reset_idx = envs.check_reset_out()
                    if len(reset_idx):
                        print(f"On step {step}, reset {reset_idx}")
                    not_reset_buf = torch.logical_not(reset_buf)
                    num_reset += len(reset_idx)

                    

                    # loss, loss_direction, loss_h, loss_ori, loss_intent = space_lossVer0(now_quad_state, predict_rel_dis, real_rel_dis, tar_pos, 7, tar_ori, criterion)
                    loss, loss_direction, loss_speed, loss_ori, loss_h = velh_lossVer5(now_quad_state, tar_pos, 7, tar_ori)
                

                    ave_loss = torch.sum(torch.mul(not_reset_buf, loss)) / (args.batch_size - len(reset_idx))
                    sum_loss += ave_loss
                    num_loss += args.batch_size - len(reset_idx)

                    now_quad_state = now_quad_state.detach()
                

                if (step + 1) % 50 == 0:
                    # print(f"Step{step}, Before: ~~~~~~~~~~~~~~~~")
                    # print("State of dynamic: ", new_state_dyn[0])
                    # print("State of simulator", new_state_sim[0]) 
                    now_quad_state = envs.reset(reset_buf=reset_buf, reset_quad_state=now_quad_state).detach()

                    ave_loss_direction = torch.sum(loss_direction) / args.batch_size
                    ave_loss_ori = torch.sum(loss_ori) / args.batch_size
                    ave_loss_h = torch.sum(loss_h) / args.batch_size
                    ave_loss = torch.sum(loss) / args.batch_size
                    
                    writer.add_scalar('Ave Loss', sum_loss / num_loss, epoch)
                    writer.add_scalar('Loss', ave_loss.item(), epoch)
                    writer.add_scalar('Loss Direction', ave_loss_direction.item(), epoch)
                    writer.add_scalar('Loss Orientation', ave_loss_ori.item(), epoch)
                    writer.add_scalar('Loss Height', ave_loss_h.item(), epoch)
                    writer.add_scalar('Number Reset', num_reset, epoch)
                    # if (step + 1) % 10 == 0:
                    #     print(f"    Step {step}: tar_pos = {tar_pos[0]}, now_pos = {now_quad_state[0, :3]}, now_evl = {now_quad_state[0, 6:9]}, action = {action[0]}")
                
                
                
                # if not step % 10:
                #     file_name = f'tmp{step}.png'
                #     envs.save_camera_output(file_name=file_name, file_path='/home/wangzimo/VTT/VTT/aerial_gym/scripts/camera_output/frames/')
            break
    print("Testing Complete!")
            

        