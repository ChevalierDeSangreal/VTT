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
import matplotlib.pyplot as plt

import pytz
from datetime import datetime
import sys
sys.path.append('/home/wangzimo/VTT/VTT')
# print(sys.path)
from aerial_gym.envs import *
from aerial_gym.utils import task_registry, velh_lossVer5, agile_lossVer1, AgileLoss, agile_loss_imageVer0
from aerial_gym.models import TrackAgileModuleVer3
from aerial_gym.envs import IsaacGymDynamics, NewtonDynamics, IsaacGymOriDynamics, NRIsaacGymDynamics
# os.path.basename(__file__).rstrip(".py")


"""
Based on trackagileVer6.py
Used to pretrain the image feature extractor
"""


def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "track_agileVer3", "help": "The name of the task."},
        {"name": "--experiment_name", "type": str, "default": "track_agileVer3", "help": "Name of the experiment to run or load."},
        {"name": "--headless", "action": "store_true", "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--num_envs", "type": int, "default": 8, "help": "Number of environments to create. Batch size will be equal to this"},
        {"name": "--seed", "type": int, "default": 42, "help": "Random seed. Overrides config file if provided."},

        # train setting
        {"name": "--learning_rate", "type":float, "default": 1.6e-6,
            "help": "the learning rate of the optimizer"},
        {"name": "--batch_size", "type":int, "default": 8,
            "help": "batch size of training. Notice that batch_size should be equal to num_envs"},
        {"name": "--num_worker", "type":int, "default": 4,
            "help": "num worker of dataloader"},
        {"name": "--num_epoch", "type":int, "default": 12520,
            "help": "num of epoch"},
        {"name": "--len_sample", "type":int, "default": 650,
            "help": "length of a sample"},
        {"name": "--tmp", "type": bool, "default": False, "help": "Set false to officially save the trainning log"},
        {"name": "--gamma", "type":int, "default": 0.8,
            "help": "how much will learning rate decrease"},
        {"name": "--slide_size", "type":int, "default": 10,
            "help": "size of GRU input window"},
        {"name": "--step_size", "type":int, "default": 100,
            "help": "learning rate will decrease every step_size steps"},

        # model setting
        {"name": "--param_save_path", "type":str, "default": '/home/wangzimo/VTT/VTT/aerial_gym/param_saved/track_agileVer7.pth',
            "help": "The path to model parameters"},
        {"name": "--param_load_path", "type":str, "default": '/home/wangzimo/VTT/VTT/aerial_gym/param_saved/track_agileVer7.pth',
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
    # torch.autograd.set_detect_anomaly(True)
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


    device = args.sim_device
    print("using device:", device)
    # print("Here I am!!!")
    envs, env_cfg = task_registry.make_env(name=args.task, args=args)
    # print("Here I am!!!")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # dynamic = IsaacGymDynamics()
    dynamic = IsaacGymDynamics()

    model = TrackAgileModuleVer3(device=device).to(device)
    # model.extractor_module.load_state_dict(torch.load('/home/wangzimo/VTT/VTT/aerial_gym/param_saved/track_agileVer7.pth', map_location=device))

    model.load_model(args.param_load_path)

    optimizer = optim.Adam(list(model.extractor_module.parameters()) + list(model.directpred.parameters()), lr=args.learning_rate, eps=1e-5)
    criterion = nn.MSELoss(reduction='none')
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    tar_ori = torch.zeros((args.batch_size, 3)).to(device)

    init_vec = torch.tensor([[1.0, 0.0, 0.0]] * args.batch_size, device=device).unsqueeze(-1)

    

    

    for epoch in range(args.num_epoch):
        
        print(f"Epoch {epoch} begin...")
        optimizer.zero_grad()


        # train
        for step in range(args.len_sample):
            now_quad_state = envs.reset(reset_buf=None).detach()

            dep_image, seg_image = envs.get_camera_dep_seg_output()
            # $$$ Why is the dep_image negative??
            dep_image = torch.abs(dep_image)
            mask = seg_image.bool()
            image_input = torch.where(mask, dep_image, torch.full_like(dep_image, 1e2))

            # # x = envs.save_camera_output(file_name="", file_path="/home/wangzimo/VTT/VTT/aerial_gym/scripts/camera_output/test_input.png", idx=0)
            # # exit(0)
            # file_path = "/home/wangzimo/VTT/VTT/aerial_gym/scripts/camera_output/test_inputVer7.png"
            # image_to_visualize = image_input[0].cpu().numpy()
            # # np.savetxt("/home/wangzimo/VTT/VTT/aerial_gym/scripts/camera_output/image_input.txt", image_to_visualize, delimiter=',', fmt='%.6f')
            # values_less_than_100 = image_to_visualize[image_to_visualize < 90]
            # # dep_image_less_than_100 = dep_image[2][dep_image[2] < 0]
            # print(values_less_than_100)
            # # print(dep_image_less_than_100)
            # # print(image_input[0])
            # plt.figure(figsize=(6, 6))
            # # plt.imshow(dep_image[2].cpu().numpy(), cmap='viridis')
            # plt.imshow(image_to_visualize, cmap='viridis')  # 可以根据需要更改 colormap
            # plt.colorbar()  # 添加颜色条以显示值范围
            # plt.title(f"Visualizing Image Input: Batch {0}")
            # plt.xlabel("X-axis")
            # plt.ylabel("Y-axis")
            # plt.savefig(file_path)
            # plt.close()
            # exit(0)
            # # print(mask[0])
            # # print(dep_image[0])
            # # print(image_input[0])

            image_input = image_input.detach()
            if torch.isnan(image_input).any():
                print("Nan detected in image_input!!!")
                exit(0)
            image_feature = model.extractor_module(image_input, mask)

            # print(image_feature[0])
            # exit(0)

            pred_dis = model.directpred(image_feature)
            # print(pred_dis[0])
            # exit(0)
            # print("Label:0.25")
            # new_state_sim, tar_state = envs.step()
            # print("Label:0.5")
            tar_state = envs.get_tar_state().detach()
            tar_pos = tar_state[:, :3].detach()
            
            now_quad_state = envs.get_quad_state().detach()

                
            loss = agile_loss_imageVer0(now_quad_state, tar_state, pred_dis)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # break

        
        writer.add_scalar('Loss', loss.item(), epoch)
            

        print(f"Epoch {epoch}, loss = {loss}")

        
        if (epoch + 1) % 50 == 0:
            print("Saving Model...")
            # model.save_model(args.param_save_path)
            # torch.save(model.extractor_module.state_dict(), args.param_save_path)
    
        # envs.update_target_traj()
    writer.close()
    print("Training Complete!")
