import os
import random
import time

import isaacgym  # noqa
from isaacgym import gymutil
from isaacgym.torch_utils import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from aerial_gym.utils import task_registry, velh_lossVer5, agile_lossVer1, AgileLoss, agile_lossVer4
from aerial_gym.models import TrackAgileModuleVer4, TrackAgileModuleVer7
from aerial_gym.envs import IsaacGymDynamics, NewtonDynamics, IsaacGymOriDynamics, NRIsaacGymDynamics
# os.path.basename(__file__).rstrip(".py")
from pytorch3d.transforms import euler_angles_to_matrix

"""
Based on trackagileVer11.py
Use new gru structure
"""


def get_args():
	custom_parameters = [
		{"name": "--task", "type": str, "default": "track_agileVer2", "help": "The name of the task."},
		{"name": "--experiment_name", "type": str, "default": "track_agileVer11", "help": "Name of the experiment to run or load."},
		{"name": "--headless", "action": "store_true", "help": "Force display off at all times"},
		{"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
		{"name": "--num_envs", "type": int, "default": 8, "help": "Number of environments to create. Batch size will be equal to this"},
		{"name": "--seed", "type": int, "default": 42, "help": "Random seed. Overrides config file if provided."},

		# train setting
		{"name": "--learning_rate", "type":float, "default": 1.6e-4,
			"help": "the learning rate of the optimizer"},
		{"name": "--batch_size", "type":int, "default": 8,
			"help": "batch size of training. Notice that batch_size should be equal to num_envs"},
		{"name": "--num_worker", "type":int, "default": 4,
			"help": "num worker of dataloader"},
		{"name": "--num_epoch", "type":int, "default": 4090,
			"help": "num of epoch"},
		{"name": "--len_sample", "type":int, "default": 1650,
			"help": "length of a sample"},
		{"name": "--tmp", "type": bool, "default": False, "help": "Set false to officially save the trainning log"},
		{"name": "--gamma", "type":int, "default": 0.8,
			"help": "how much will learning rate decrease"},
		{"name": "--slide_size", "type":int, "default": 10,
			"help": "size of GRU input window"},
		{"name": "--step_size", "type":int, "default": 100,
			"help": "learning rate will decrease every step_size steps"},

		# model setting
		{"name": "--param_save_path", "type":str, "default": '/home/wangzimo/VTT/VTT/aerial_gym/param_saved/track_agileVer15.pth',
			"help": "The path to model parameters"},
		{"name": "--param_load_path", "type":str, "default": '/home/wangzimo/VTT/VTT/aerial_gym/param_saved/track_agileVer15.pth',
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

	# dynamic = IsaacGymDynamics()
	dynamic = IsaacGymDynamics()

	# tmp_model = TrackAgileModuleVer3(device=device).to(device)
	model = TrackAgileModuleVer7(device=device).to(device)

	model.load_model(args.param_load_path)
	# tmp_model.load_model(args.param_load_path)
	# model.directpred.load_state_dict(tmp_model.directpred.state_dict())
	# model.extractor_module.load_state_dict(torch.load('/home/wangzimo/VTT/VTT/aerial_gym/param_saved/track_agileVer7.pth', map_location=device))



	# optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
	criterion = nn.MSELoss(reduction='none')
	# scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

	tar_ori = torch.zeros((args.batch_size, 3)).to(device)

	init_vec = torch.tensor([[1.0, 0.0, 0.0]] * args.batch_size, device=device).unsqueeze(-1)

	model.eval()

	
	with torch.no_grad():
		for epoch in range(args.num_epoch):
			
			print(f"Epoch {epoch} begin...")
			old_loss = AgileLoss(args.batch_size, device=device)
			optimizer.zero_grad()
			
			timer = torch.zeros((args.batch_size,), device=device)
			h0 = None
			reset_buf = None
			now_quad_state = envs.reset(reset_buf=reset_buf).detach()
			
			if torch.isnan(now_quad_state[:, 3:6]).any():
				# print(input_buffer[max(step+1-args.slide_size, 0):step+1])
				print("Nan detected in early input!!!")
				exit(0)
			reset_buf = torch.zeros((args.batch_size,))
			
			num_reset = 0
			tar_state = envs.get_tar_state().detach()
			# train
			for step in range(args.len_sample):
				
				# # rel_dis = envs.get_relative_distance()
				rel_dis = tar_state[:, :3] - now_quad_state[:, :3]
				# rel_dis_3 = rel_dis.repeat(1, 3)

				# # ----------------------------
				# dep_image = torch.abs(dep_image)
				# mask = seg_image.bool()
				# image_input = torch.where(mask, dep_image, torch.full_like(dep_image, 1e2))
				# image_input = image_input.detach()
				# image_feature = model.extractor_module(image_input, mask)
				# # ----------------------------

				world_to_body = dynamic.world_to_body_matrix(now_quad_state[:, 3:6].detach())
				body_to_world = torch.transpose(world_to_body, 1, 2)
				
				body_rel_dis = torch.matmul(world_to_body, torch.unsqueeze(rel_dis, 2)).squeeze(-1)
				body_vel = torch.matmul(world_to_body, torch.unsqueeze(now_quad_state[:, 6:9], 2)).squeeze(-1)
				body_acc = torch.matmul(world_to_body, torch.unsqueeze(now_quad_state[:, 9:12], 2)).squeeze(-1)
				tmp_input = torch.cat((body_vel, body_acc, now_quad_state[:, 3:6], body_rel_dis), dim=1)
				
				# body_pred_dis = model.directpred(tmp_input_directpred)


				tmp_input = tmp_input.unsqueeze(0)

				action, hn = model.decision_module(tmp_input, h0)
				# print(tmp_input[0][0], action[0])
				# exit(0)
				# print("Label:0")
				new_state_dyn, acceleration = dynamic(now_quad_state, action, envs.cfg.sim.dt)
				# print("Label:0.25")
				new_state_sim, tar_state = envs.step(new_state_dyn.detach())
				# print("Label:0.5")
				x = envs.save_camera_output(file_name=f'{step}.png', idx=3)
				tar_pos = tar_state[:, :3].detach()
				
				now_quad_state = new_state_dyn

				# print("Label:1")
				reset_buf, reset_idx = envs.check_reset_out()
				# if len(reset_idx):
				#     print(f"On step {step}, reset {reset_idx}")
				not_reset_buf = torch.logical_not(reset_buf)
				num_reset += len(reset_idx)

				# world_to_body = dynamic.world_to_body_matrix(now_quad_state[:, 3:6].detach())
				# body_to_world = torch.transpose(world_to_body, 1, 2)
				# world_pred_dis = torch.matmul(body_to_world, torch.unsqueeze(body_pred_dis, 2)).squeeze(-1)
				# loss, new_loss = agile_lossVer5(old_loss, now_quad_state, tar_state, 7, tar_ori, 3, timer, envs.cfg.sim.dt, init_vec, world_pred_dis)
				loss, new_loss = agile_lossVer4(old_loss, now_quad_state, tar_state, 7, tar_ori, 2, timer, envs.cfg.sim.dt, init_vec)
				old_loss = new_loss
				# print("Label:2")
				
				# loss.backward(reset_buf, retain_graph=True)
				now_quad_state[reset_idx] = envs.reset(reset_buf=reset_buf)[reset_idx].detach()
				rotation_matrices = euler_angles_to_matrix(now_quad_state[:, 3:6], convention='XYZ')
				direction_vector = rotation_matrices @ init_vec
				direction_vector = direction_vector.squeeze()

				cos_sim = F.cosine_similarity(direction_vector, rel_dis, dim=1)
				theta = torch.acos(cos_sim)
				theta_degrees = theta * 180.0 / torch.pi

				cos_sim_hor = F.cosine_similarity(direction_vector[:, :2], rel_dis[:, :2], dim=1)
				theta_hor = torch.acos(cos_sim_hor)
				theta_degrees_hor = theta_hor * 180.0 / torch.pi
				
				item_tested = 3
				horizon_dis = torch.norm(now_quad_state[item_tested, :2] - tar_pos[item_tested, :2], dim=0, p=4)
				speed = torch.norm(now_quad_state[item_tested, 6:9], dim=0, p=2)

				if reset_buf[item_tested]:
					loss[item_tested] = float('nan')
				writer.add_scalar(f'Total Loss', loss[item_tested], step)
				writer.add_scalar(f'Direction Loss/sum', old_loss.direction[item_tested], step)
				writer.add_scalar(f'Direction Loss/xy', old_loss.direction_hor[item_tested], step)
				writer.add_scalar(f'Direction Loss/z', old_loss.direction_ver[item_tested], step)
				writer.add_scalar(f'Distance Loss', old_loss.distance[item_tested], step)
				writer.add_scalar(f'Velocity Loss', old_loss.vel[item_tested], step)
				writer.add_scalar(f'Orientation Loss', old_loss.ori[item_tested], step)
				writer.add_scalar(f'Height Loss', old_loss.h[item_tested], step)

				writer.add_scalar(f'Orientation/X', direction_vector[item_tested, 0], step)
				writer.add_scalar(f'Orientation/Y', direction_vector[item_tested, 1], step)
				writer.add_scalar(f'Orientation/Z', direction_vector[item_tested, 2], step)
				writer.add_scalar(f'Orientation/Theta', theta_degrees[item_tested], step)
				writer.add_scalar(f'Orientation/ThetaXY', theta_degrees_hor[item_tested], step)
				writer.add_scalar(f'Acceleration/X', acceleration[item_tested, 0], step)
				writer.add_scalar(f'Acceleration/Y', acceleration[item_tested, 1], step)
				writer.add_scalar(f'Acceleration/Z', acceleration[item_tested, 2], step)
				writer.add_scalar(f'Horizon Distance', horizon_dis, step)
				writer.add_scalar(f'Position/X', now_quad_state[item_tested, 0], step)
				writer.add_scalar(f'Position/Y', now_quad_state[item_tested, 1], step)
				writer.add_scalar(f'Target Position/X', tar_pos[item_tested, 0], step)
				writer.add_scalar(f'Target Position/Y', tar_pos[item_tested, 1], step)
				writer.add_scalar(f'Velocity/X', now_quad_state[item_tested, 6], step)
				writer.add_scalar(f'Velocity/Y', now_quad_state[item_tested, 7], step)
				writer.add_scalar(f'Distance/X', tar_pos[item_tested, 0] - now_quad_state[item_tested, 0], step)
				writer.add_scalar(f'Distance/Y', tar_pos[item_tested, 1] - now_quad_state[item_tested, 1], step)
				writer.add_scalar(f'Action/F', action[item_tested, 0], step)
				writer.add_scalar(f'Action/X', action[item_tested, 1], step)
				writer.add_scalar(f'Action/Y', action[item_tested, 2], step)
				writer.add_scalar(f'Action/Z', action[item_tested, 3], step)
				writer.add_scalar(f'Speed/Z', now_quad_state[item_tested, 8], step)
				writer.add_scalar(f'Speed', speed, step)
				writer.add_scalar(f'Height', now_quad_state[item_tested, 2], step)

				
				old_loss.reset(reset_idx=reset_idx)
				h0 = hn
				h0[:, reset_idx] = 0
				timer = timer + 1
				timer[reset_idx] = 0



			ave_loss_direciton = torch.sum(new_loss.direction) / args.batch_size
			ave_loss_distance = torch.sum(new_loss.distance) / args.batch_size
			ave_loss_velocity = torch.sum(new_loss.vel) / args.batch_size
			ave_loss_ori = torch.sum(new_loss.ori) / args.batch_size
			ave_loss_h = torch.sum(new_loss.h) / args.batch_size
			ave_loss_aux = torch.sum(new_loss.aux) / args.batch_size
			# ave_loss_intent = torch.sum(loss_intent) / args.batch_size
			ave_loss = torch.sum(loss) / args.batch_size
			
			writer.add_scalar('Loss', ave_loss.item(), epoch)
			writer.add_scalar('Loss Direction', ave_loss_direciton.item(), epoch)
			writer.add_scalar('Loss Distance', ave_loss_distance.item(), epoch)
			writer.add_scalar('Loss Velocity', ave_loss_velocity.item(), epoch)
			# writer.add_scalar('Loss Intent', ave_loss_intent.item(), epoch)
			writer.add_scalar('Loss Orientation', ave_loss_ori.item(), epoch)
			writer.add_scalar('Loss Height', ave_loss_h.item(), epoch)
			writer.add_scalar('Loss Aux', ave_loss_aux.item(), epoch)
			writer.add_scalar('Number Reset', num_reset, epoch)
				

			print(f"Epoch {epoch}, Ave loss = {ave_loss}, num reset = {num_reset}")
			break
			

			# envs.update_target_traj()
		writer.close()
		print("Testing Complete!")
