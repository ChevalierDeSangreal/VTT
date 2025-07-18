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
from aerial_gym.utils import task_registry, velh_lossVer5, agile_lossVer1, AgileLoss, agile_lossVer4
from aerial_gym.models import TrackTransferModuleVer0
from aerial_gym.envs import IsaacGymDynamics, NewtonDynamics, IsaacGymOriDynamics, NRIsaacGymDynamics
# os.path.basename(__file__).rstrip(".py")


"""
Based on trackagileVer11.py
"""


def get_args():
	custom_parameters = [
		{"name": "--task", "type": str, "default": "track_agileVer2", "help": "The name of the task."},
		{"name": "--experiment_name", "type": str, "default": "track_transferVer0", "help": "Name of the experiment to run or load."},
		{"name": "--headless", "action": "store_true", "help": "Force display off at all times"},
		{"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
		{"name": "--num_envs", "type": int, "default": 32, "help": "Number of environments to create. Batch size will be equal to this"},
		{"name": "--seed", "type": int, "default": 42, "help": "Random seed. Overrides config file if provided."},

		# train setting
		{"name": "--learning_rate", "type":float, "default": 1.6e-4,
			"help": "the learning rate of the optimizer"},
		{"name": "--batch_size", "type":int, "default": 32,
			"help": "batch size of training. Notice that batch_size should be equal to num_envs"},
		{"name": "--num_worker", "type":int, "default": 4,
			"help": "num worker of dataloader"},
		{"name": "--num_epoch", "type":int, "default": 4090,
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
		{"name": "--param_save_path", "type":str, "default": '/home/wangzimo/VTT/VTT/aerial_gym/param/track_transferVer0.pth',
			"help": "The path to model parameters"},
		{"name": "--param_load_path", "type":str, "default": '/home/wangzimo/VTT/VTT/aerial_gym/param/track_transferVer0.pth',
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

	# tmp_model = TrackAgileModuleVer3(device=device).to(device)
	model = TrackTransferModuleVer0(device=device).to(device)

	# model.load_model(args.param_load_path)
	# tmp_model.load_model(args.param_load_path)
	# model.directpred.load_state_dict(tmp_model.directpred.state_dict())
	# model.extractor_module.load_state_dict(torch.load('/home/wangzimo/VTT/VTT/aerial_gym/param_saved/track_agileVer7.pth', map_location=device))

	for name, param in model.named_parameters():
		if ("extractor_module" in name) or ("directpred" in name):
			param.requires_grad = False


	# optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
	criterion = nn.MSELoss(reduction='none')
	# scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

	tar_ori = torch.zeros((args.batch_size, 3)).to(device)

	init_vec = torch.tensor([[1.0, 0.0, 0.0]] * args.batch_size, device=device).unsqueeze(-1)

	

	

	for epoch in range(args.num_epoch):
		
		print(f"Epoch {epoch} begin...")
		old_loss = AgileLoss(args.batch_size, device=device)
		optimizer.zero_grad()
		
		timer = torch.zeros((args.batch_size,), device=device)

		input_buffer = torch.zeros(args.slide_size, args.batch_size, 9+3).to(device)
		predict_buffer = torch.zeros(args.batch_size, 5, 9).to(device)

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


			
			if step % 10 == 0:
				rel_dis = tar_state[:, :3] - now_quad_state[:, :3]
				world_to_body = dynamic.world_to_body_matrix(now_quad_state[:, 3:6].detach())
				body_to_world = torch.transpose(world_to_body, 1, 2)

				body_rel_dis = torch.matmul(world_to_body, torch.unsqueeze(rel_dis, 2)).squeeze(-1)
				body_vel = torch.matmul(world_to_body, torch.unsqueeze(now_quad_state[:, 6:9], 2)).squeeze(-1)
				body_acc = torch.matmul(world_to_body, torch.unsqueeze(now_quad_state[:, 9:12], 2)).squeeze(-1)
				
				tmp_input = torch.cat((body_vel, body_acc, now_quad_state[:, 3:6], body_rel_dis), dim=1)
			
				tmp_input = tmp_input.unsqueeze(0)
				input_buffer = input_buffer[1:].clone()
				input_buffer = torch.cat((input_buffer, tmp_input), dim=0)
				
				action_seq, embedding = model.decision_module(input_buffer.clone())
				
				if step % 50 == 0:
					init_pos = now_quad_state[:, :3].detach()
					# print("embedding shape:", embedding.shape)
					predict_res = model.predict_module(embedding)
			
				idx_predict = (step % 50) // 10
				predict_buffer[:, idx_predict, :3] = now_quad_state[:, :3].detach() - init_pos
				predict_buffer[:, idx_predict, 3:6] = body_vel.detach()
				predict_buffer[:, idx_predict, 6:9] = now_quad_state[:, 3:6].detach()


			# action: [batch_size, 10, 4]
			action = action_seq[:, step % 10, :].clone()
			
			new_state_dyn, acceleration = dynamic(now_quad_state, action, envs.cfg.sim.dt)
			new_state_sim, tar_state = envs.step(new_state_dyn.detach())
			tar_pos = tar_state[:, :3].detach()
			
			now_quad_state = new_state_dyn

			reset_buf, reset_idx = envs.check_reset_out()
			not_reset_buf = torch.logical_not(reset_buf)
			num_reset += len(reset_idx)
			input_buffer[:, reset_idx] = 0

			loss_agile, new_loss = agile_lossVer4(old_loss, now_quad_state, tar_state, 7, tar_ori, 2, timer, envs.cfg.sim.dt, init_vec)
			old_loss = new_loss
			
			
			now_quad_state[reset_idx] = envs.reset(reset_buf=reset_buf)[reset_idx].detach()
			old_loss.reset(reset_idx=reset_idx)
			timer = timer + 1
			timer[reset_idx] = 0
			# print("Length of reset buf:", len(reset_idx), not_reset_buf)

			if (not (step + 1) % 50):
				
				# print(action[0])
				# print("Loss:", loss[0])
				# print("shape of predict buffer:", predict_buffer.shape)
				# print("Shape of predict res:", predict_res.shape)
				loss_pred = criterion(predict_buffer, predict_res).mean(dim=(1, 2))
				# print("Loss pred shape:", loss_pred.shape)
				# print("Loss agile shape:", loss_agile.shape)
				loss = 0.1 * loss_pred + 0.9 * loss_agile
				loss.backward(not_reset_buf)
				optimizer.step()
				optimizer.zero_grad()
				now_quad_state = now_quad_state.detach()
				old_loss = AgileLoss(args.batch_size, device=device)
				input_buffer = input_buffer.detach()
				timer = timer * 0
				predict_buffer.zero_().detach_()



		ave_loss_direciton = torch.sum(new_loss.direction) / args.batch_size
		ave_loss_distance = torch.sum(new_loss.distance) / args.batch_size
		ave_loss_velocity = torch.sum(new_loss.vel) / args.batch_size
		ave_loss_ori = torch.sum(new_loss.ori) / args.batch_size
		ave_loss_h = torch.sum(new_loss.h) / args.batch_size
		# ave_loss_aux = torch.sum(new_loss.aux) / args.batch_size
		# ave_loss_intent = torch.sum(loss_intent) / args.batch_size
		ave_loss_predict = torch.sum(loss_pred) / args.batch_size
		ave_loss_agile = torch.sum(loss_agile) / args.batch_size
		ave_loss = torch.sum(loss) / args.batch_size
		
		writer.add_scalar('Loss', ave_loss.item(), epoch)
		writer.add_scalar('Loss Direction', ave_loss_direciton.item(), epoch)
		writer.add_scalar('Loss Distance', ave_loss_distance.item(), epoch)
		writer.add_scalar('Loss Velocity', ave_loss_velocity.item(), epoch)
		# writer.add_scalar('Loss Intent', ave_loss_intent.item(), epoch)
		writer.add_scalar('Loss Orientation', ave_loss_ori.item(), epoch)
		writer.add_scalar('Loss Height', ave_loss_h.item(), epoch)
		# writer.add_scalar('Loss Aux', ave_loss_aux.item(), epoch)
		writer.add_scalar('Loss Predict', ave_loss_predict.item(), epoch)
		writer.add_scalar('Loss Agile', ave_loss_agile.item(), epoch)
		writer.add_scalar('Number Reset', num_reset, epoch)
			

		print(f"Epoch {epoch}, Ave loss = {ave_loss}, num reset = {num_reset}")

		
		if epoch == 2000:  
			for param_group in optimizer.param_groups:
				param_group['lr'] = 1.6e-5
		
		if (epoch + 1) % 4000 == 0:
			print("Saving Model...")
			model.save_model(args.param_save_path)
	
		# envs.update_target_traj()
	writer.close()
	print("Training Complete!")
