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
from torch.utils.data import Dataset, DataLoader
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import pytz
from datetime import datetime
import sys
sys.path.append('/home/wangzimo/VTT/VTT')
# print(sys.path)
from aerial_gym.envs import *
from aerial_gym.utils import task_registry, velh_lossVer5, agile_lossVer1, AgileLoss, agile_lossVer4
from aerial_gym.models import TrackTransferModuleVer0, WorldModelVer0
from aerial_gym.envs import IsaacGymDynamics, NewtonDynamics, IsaacGymOriDynamics, NRIsaacGymDynamics
# os.path.basename(__file__).rstrip(".py")

from tqdm import tqdm
"""
Based on tracktransferVer1.py
Train a world model and a controller
"""


def get_args():
	custom_parameters = [
		{"name": "--task", "type": str, "default": "track_agileVer2", "help": "The name of the task."},
		{"name": "--experiment_name", "type": str, "default": "track_transferVer2", "help": "Name of the experiment to run or load."},
		{"name": "--headless", "action": "store_true", "help": "Force display off at all times"},
		{"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
		{"name": "--num_envs", "type": int, "default": 32, "help": "Number of environments to create. Batch size will be equal to this"},
		{"name": "--seed", "type": int, "default": 42, "help": "Random seed. Overrides config file if provided."},

		# train setting
		{"name": "--learning_rate", "type":float, "default": 1.6e-5,
			"help": "the learning rate of the optimizer"},
		{"name": "--batch_size", "type":int, "default": 32,
			"help": "batch size of training. Notice that batch_size should be equal to num_envs"},
		{"name": "--num_worker", "type":int, "default": 4,
			"help": "num worker of dataloader"},
		{"name": "--num_epoch", "type":int, "default": 50,
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
		{"name": "--param_save_path", "type":str, "default": '/home/wangzimo/VTT/VTT/aerial_gym/param/track_transferVer2.pth',
			"help": "The path to model parameters"},
		{"name": "--param_load_path", "type":str, "default": '/home/wangzimo/VTT/VTT/aerial_gym/param_saved/track_transferVer0.pth',
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

class EmbStateDataset(Dataset):
	def __init__(self, data_dir='/home/wangzimo/VTT/data', device='cpu'):
		self.device = device
		self.embeddings = []
		self.states = []

		# 遍历所有 pt 文件并加载数据
		for filename in sorted(os.listdir(data_dir)):
			if filename.endswith('.pt'):
				file_path = os.path.join(data_dir, filename)
				data = torch.load(file_path, map_location='cpu')
				self.embeddings.append(data['embeddings'])  # (N, 256)
				self.states.append(data['states'])          # (N, 12)
			break

		# 拼接所有文件中的数据
		self.embedding = torch.cat(self.embeddings, dim=0)
		self.state = torch.cat(self.states, dim=0)

	def __len__(self):
		return self.embedding.shape[0]

	def __getitem__(self, idx):
		emb = self.embedding[idx]
		stt = self.state[idx]
		return emb, stt


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
	# print("Here I am!!!")
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	# dynamic = IsaacGymDynamics()
	dynamic = IsaacGymDynamics()

	model_actor = TrackTransferModuleVer0(device=device).to(device)
	model_world = WorldModelVer0(device=device).to(device)

	model_actor.load_model(args.param_load_path)

	criterion = nn.MSELoss()

	dataset = EmbStateDataset(device=device)
	dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)


	for epoch in range(args.num_epoch):

		print(f"Epoch {epoch} begin...")
		# for batch_idx, (embedding, state) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epoch}")):
		for batch_idx, (embedding, state) in enumerate(dataloader):
			embedding = embedding.to(device)
			state = state.to(device)

			loss_world_model = 0.0
			loss_actor = 0.0

			now_quad_state = state.to(device)
			world_model_quad_state = now_quad_state.clone()

			out = model_actor.decision_module.fc(embedding)
			out = torch.sigmoid(out) * 2 - 1
			action_seq = out.view(args.batch_size, args.slide_size, -1)
			# print("Here I am!!!")
			# print(action_seq.shape)

			trajectory_decoder = model_actor.predict_module(embedding)
			init_pos = now_quad_state[:, :3].clone()
			for step in range(args.slide_size):
				action = action_seq[:, step, :].clone()

				# print("action shape:", action.shape)
				# print("now_quad_state shape:", now_quad_state.shape)


				world_to_body = dynamic.world_to_body_matrix(now_quad_state[:, 3:6])
				body_to_world = torch.transpose(world_to_body, 1, 2)
				body_vel = torch.matmul(world_to_body, torch.unsqueeze(now_quad_state[:, 6:9], 2)).squeeze(-1)
				tmp_delta_pos = now_quad_state[:, :3] - init_pos
				tmp_vel_body = body_vel
				tmp_angles_body = now_quad_state[:, 3:6]
				tmp_state_dyn = torch.cat([tmp_delta_pos, tmp_vel_body, tmp_angles_body], dim=1)

				# loss_actor += criterion(trajectory_decoder[:, step, :], world_predict)
				loss_actor += criterion(trajectory_decoder[:, step, :], tmp_state_dyn.detach())
				print("trajactory_decoder", trajectory_decoder[0, step, :])
				print("state_dyn", tmp_state_dyn[0])
				
				new_state_dyn, acceleration = dynamic(now_quad_state, action, 0.02)
				now_quad_state = new_state_dyn

			break

			writer.add_scalar("loss/actor_loss", loss_actor, batch_idx)
			writer.add_scalar("loss/world_loss", loss_world_model, batch_idx)
			writer.add_scalar("loss/loss", loss, batch_idx)
		# writer.add_scalar("loss/actor_loss", loss_actor, epoch)
		# writer.add_scalar("loss/world_loss", loss_world_model, epoch)
		# writer.add_scalar("loss/loss", loss, epoch)

		break




	writer.close()
	print("Training Complete!")
