# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
""" 
    Modified based on track_agileVer1.py
    Using image as input
    $$$ By now a moving target is set
    $$$ Two place to change from 12 to 13
"""
import numpy as np
import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch.utils.data import DataLoader
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_euler_angles, euler_angles_to_matrix, matrix_to_quaternion, axis_angle_to_matrix


from aerial_gym import AERIAL_GYM_ROOT_DIR, AERIAL_GYM_ROOT_DIR

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import *
from aerial_gym.envs.base.base_task import BaseTask
from .track_space_config import TrackSpaceCfg
from .track_agile_config import TrackAgileCfg

from aerial_gym.utils.helpers import asset_class_to_AssetOptions
from aerial_gym.utils.mymath import rand_circle_point

from aerial_gym.data.dataset import TargetDataset
from aerial_gym.data.generate_dataVer2 import TrajectoryGenerator

from aerial_gym.envs.base.dynamics_isaac import IsaacGymDynamics
import torch.nn.functional as F
import itertools

class TrackAgileVer2(BaseTask):

    def __init__(self, cfg: TrackAgileCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg

        self.max_episode_length = int(self.cfg.env.episode_length_s / self.cfg.sim.dt)
        self.max_len_sample = self.cfg.env.max_sample_length
        self.debug_viz = False
        num_actors = 3

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device_id = sim_device
        self.headless = headless
        

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.tar_obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)

        bodies_per_env = self.robot_num_bodies + self.tar_num_bodies

        self.vec_root_tensor = gymtorch.wrap_tensor(
            self.root_tensor).view(self.num_envs, num_actors, 13)

        self.root_states = self.vec_root_tensor[:, self.robot_actor_index, :]
        self.root_positions = self.root_states[:, 0:3]
        self.root_quats = self.root_states[:, 3:7]
        self.root_linvels = self.root_states[:, 7:10]
        self.root_angvels = self.root_states[:, 10:13]

        self.tar_root_states = self.vec_root_tensor[:, self.tar_actor_index, :]
        self.tar_root_positions = self.tar_root_states[:, 0:3]
        self.tar_root_quats = self.tar_root_states[:, 3:7]
        self.tar_root_linvels = self.tar_root_states[:, 7:10]
        self.tar_root_angvels = self.tar_root_states[:, 10:13]

        self.privileged_obs_buf = None
        if self.vec_root_tensor.shape[1] > 1:
            self.env_asset_root_states = self.vec_root_tensor[:, 1:, :]
            if self.get_privileged_obs:
                self.privileged_obs_buf = self.env_asset_root_states

        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.initial_root_states = self.root_states.clone()
        self.initial_tar_states = self.tar_root_states.clone()
        self.counter = 0

        self.action_upper_limits = torch.tensor(
            [1, 1, 1, 1], device=self.device, dtype=torch.float32)
        self.action_lower_limits = torch.tensor(
            [-1, -1, -1, -1], device=self.device, dtype=torch.float32)

        # control tensors
        self.action_input = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device, requires_grad=False)
        self.forces = torch.zeros((self.num_envs, bodies_per_env, 3),
                                  dtype=torch.float32, device=self.device, requires_grad=False)
        self.torques = torch.zeros((self.num_envs, bodies_per_env, 3),
                                   dtype=torch.float32, device=self.device, requires_grad=False)

        # self.controller = Controller(self.cfg.control, self.device)
        self.dynamics = IsaacGymDynamics()

        if self.viewer:
            cam_pos_x, cam_pos_y, cam_pos_z = self.cfg.viewer.pos[0], self.cfg.viewer.pos[1], self.cfg.viewer.pos[2]
            cam_target_x, cam_target_y, cam_target_z = self.cfg.viewer.lookat[0], self.cfg.viewer.lookat[1], self.cfg.viewer.lookat[2]
            cam_pos = gymapi.Vec3(cam_pos_x, cam_pos_y, cam_pos_z)
            cam_target = gymapi.Vec3(cam_target_x, cam_target_y, cam_target_z)
            cam_ref_env = self.cfg.viewer.ref_env
            
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # # init dataset
        # self.tar_traj_dataset = TargetDataset('/home/wangzimo/VTT/VTT/aerial_gym/data', self.device)
        # self.tar_traj_dataloader = DataLoader(self.tar_traj_dataset, batch_size=self.num_envs, shuffle=True)
        # self.tar_traj_iter = itertools.cycle(self.tar_traj_dataloader)
        # self.tar_traj = next(self.tar_traj_iter)
        # # print("Shape of tar_traj:", self.tar_traj.shape)
        # init data generator
        
        # tar_v = 5
        # direction_change_interval = 3
        # self.traj_generator = TrajectoryGenerator(tar_v, self.cfg.sim.dt, direction_change_interval, total_time=10, batch_size=self.num_envs, device=self.device)
        # self.traj = self.traj_generator.batch_generate_trajectories()

        self.tar_acc_norm = 2
        self.tar_acc_intervel = 100 # How many time steps will acceleration change once
        self.tar_acc = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)

        self.count_step = torch.zeros((self.num_envs, ), dtype=torch.long, device=self.device)

        self.initial_vector = torch.tensor([[1.0, 0.0, 0.0]] * self.num_envs, device=self.device)

    def create_sim(self):
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        
        self._create_ground_plane()
        
        self._create_envs()
        
        self.progress_buf = torch.zeros(
            self.cfg.env.num_envs, device=self.sim_device, dtype=torch.long)
        

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        return

    def _create_envs(self):
        print("\n\n\n\n\n CREATING ENVIRONMENT \n\n\n\n\n\n")
        asset_path = self.cfg.robot_asset.file.format(
            AERIAL_GYM_ROOT_DIR=AERIAL_GYM_ROOT_DIR)
        tar_asset_path = self.cfg.tar_asset.file.format(
            AERIAL_GYM_ROOT_DIR=AERIAL_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        tar_asset_root = os.path.dirname(tar_asset_path)
        tar_asset_file = os.path.basename(tar_asset_path)

        asset_options = asset_class_to_AssetOptions(self.cfg.robot_asset)
        # asset_options.override_inertia = 
        tar_asset_options = asset_class_to_AssetOptions(self.cfg.tar_asset)

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options)

        tar_asset = self.gym.load_asset(
            self.sim, tar_asset_root, tar_asset_file, tar_asset_options)

        self.robot_num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.tar_num_bodies = self.gym.get_asset_rigid_body_count(tar_asset)

        start_pose = gymapi.Transform()
        self.env_spacing = self.cfg.env.env_spacing
        env_lower = gymapi.Vec3(-self.env_spacing, -
                                self.env_spacing, -self.env_spacing)
        env_upper = gymapi.Vec3(
            self.env_spacing, self.env_spacing, self.env_spacing)
        self.actor_handles = []
        self.camera_handles = []
        self.camera2_handles = []
        self.envs = []
        self.camera_root_tensors = []
        self.camera_dep_root_tensors = []
        self.camera_seg_root_tensors = []
        self.tar_seg_ids = []
        tar_seg_id_count = 1
        
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            # print("??????: 1")
            pos = torch.tensor([0, 0, 3], device=self.device)
            start_pose.p = gymapi.Vec3(*pos)
            actor_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose, self.cfg.robot_asset.name, i, self.cfg.robot_asset.collision_mask, 0)
            self.robot_actor_index = self.gym.get_actor_index(env_handle, actor_handle, gymapi.IndexDomain.DOMAIN_ENV)
            self.robot_body_index = self.gym.get_actor_rigid_body_index(env_handle, actor_handle, 0, gymapi.IndexDomain.DOMAIN_ENV)
            self.robot_body_handle = self.gym.get_actor_rigid_body_handle(env_handle, actor_handle, 0)




            # camera_offset = gymapi.Vec3(0.21, 0, 0.05)
            # camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(0))
            # # print("??????: 2.6")
            # camera2_handle = self.gym.create_camera_sensor(env_handle, camera_properties)
            # self.gym.attach_camera_to_body(camera2_handle, env_handle, self.robot_body_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_TRANSFORM)
            # # self.gym.set_camera_location(camera_handle, env_handle, gymapi.Vec3(1,1,1), gymapi.Vec3(1,1,2))
            # camera2_handle = self.gym.create_camera_sensor(env_handle, camera_properties)
            # self.camera2_handles.append(camera2_handle)


            
            # print("??????: 4")
            pos = torch.tensor([0, 0, 0.2], device=self.device)
            start_pose.p = gymapi.Vec3(*pos)
            tar_actor_handle = self.gym.create_actor(
                env_handle, tar_asset, start_pose, self.cfg.tar_asset.name, i, self.cfg.robot_asset.collision_mask, 0)
            self.tar_body_handle = self.gym.get_actor_rigid_body_handle(env_handle, tar_actor_handle, 0)
            self.tar_actor_index = self.gym.get_actor_index(env_handle, tar_actor_handle, gymapi.IndexDomain.DOMAIN_ENV)
            self.tar_body_index = self.gym.get_actor_rigid_body_index(env_handle, tar_actor_handle, 0, gymapi.IndexDomain.DOMAIN_ENV)

            pos = torch.tensor([0, 0, -1.2], device=self.device)
            start_pose.p = gymapi.Vec3(*pos)
            tmp_actor_handle = self.gym.create_actor(
                env_handle, tar_asset, start_pose, self.cfg.tar_asset.name, i, self.cfg.robot_asset.collision_mask, 0)
            self.tmp_body_handle = self.gym.get_actor_rigid_body_handle(env_handle, tmp_actor_handle, 0)
            # print("??????: 5")
            tar_seg_id_count += 1
            self.gym.set_rigid_body_segmentation_id(env_handle, tar_actor_handle, 0, tar_seg_id_count)
            self.tar_seg_ids.append(tar_seg_id_count)
            self.robot_body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.actor_handles.append(tar_actor_handle)
            # print("??????: 6")

            # print("??????: 2")
            # Create Cemara
            camera_properties = gymapi.CameraProperties()
            camera_properties.width = 224
            camera_properties.height = 224
            camera_properties.enable_tensors = True


            
            # camera_handle = self.gym.create_camera_sensor(env_handle, camera_properties)
            # # print("??????: 2.5")
            # camera_offset = gymapi.Vec3(0, 0, 13.05)
            # camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(90))
            # # print("??????: 2.6")
            # self.gym.attach_camera_to_body(camera_handle, env_handle, self.tmp_body_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_POSITION)
            # # self.gym.set_camera_location(camera_handle, env_handle, gymapi.Vec3(0,0,12), gymapi.Vec3(0,0,0))
            # self.camera_handles.append(camera_handle)
            # # print("??????: 3")
            # camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, camera_handle, gymapi.IMAGE_COLOR)
            # torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)
            # self.camera_root_tensors.append(torch_camera_tensor)

            # camera_dep_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, camera_handle, gymapi.IMAGE_SEGMENTATION)
            # torch_camera_dep_tensor = gymtorch.wrap_tensor(camera_dep_tensor)
            # self.camera_dep_root_tensors.append(torch_camera_dep_tensor)

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_properties)
            # print("??????: 2.5")
            camera_offset = gymapi.Vec3(0.21, 0, 0.05)
            camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(0))
            # print(camera_rotation)
            # exit(0)
            # print("??????: 2.6")
            self.gym.attach_camera_to_body(camera_handle, env_handle, self.robot_body_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_TRANSFORM)
            # self.gym.set_camera_location(camera_handle, env_handle, gymapi.Vec3(0,0,12), gymapi.Vec3(0,0,0))
            self.camera_handles.append(camera_handle)
            # print("??????: 3")
            camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, camera_handle, gymapi.IMAGE_COLOR)
            torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)
            self.camera_root_tensors.append(torch_camera_tensor)

            camera_dep_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, camera_handle, gymapi.IMAGE_DEPTH)
            torch_camera_dep_tensor = gymtorch.wrap_tensor(camera_dep_tensor)
            self.camera_dep_root_tensors.append(torch_camera_dep_tensor)

            camera_seg_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, camera_handle, gymapi.IMAGE_SEGMENTATION)
            torch_camera_seg_tensor = gymtorch.wrap_tensor(camera_seg_tensor)
            self.camera_seg_root_tensors.append(torch_camera_seg_tensor)

        self.robot_mass = 0
        for prop in self.robot_body_props:
            self.robot_mass += prop.mass
            # print(dir(prop.inertia))
            # print("Inertia: ", prop.inertia.x, prop.inertia.y, prop.inertia.z, prop.mass)
        print("Total robot mass: ", self.robot_mass)
        
        print("\n\n\n\n\n ENVIRONMENT CREATED \n\n\n\n\n\n")

    def step(self, new_states):

        # step physics and render each frame
        for i in range(self.cfg.env.num_control_steps_per_env_step):
            # print("##### 3")
            self.pre_physics_step(new_states)
            self.gym.simulate(self.sim)
            # print("##### 4")
            # NOTE: as per the isaacgym docs, self.gym.fetch_results must be called after self.gym.simulate, but not having it here seems to work fine
            # it is called in the render function.
            self.post_physics_step()
            # print("##### 5")

        # set position
        # self.tar_root_states[range(self.num_envs), 0:3] = self.tar_traj[range(self.num_envs), self.count_step[range(self.num_envs)], :3]
        # self.tar_root_states[:, 2] = 0.2
        # # set linearvels
        # self.tar_root_states[:, 7:10] = self.tar_traj[range(self.num_envs), self.count_step[range(self.num_envs)], 6:9]
        # # set angvels
        # self.tar_root_states[:, 10:13] = 0
        # # set quats
        # self.tar_root_states[:, 3:7] = 0
        # self.tar_root_states[:, 6] = 2
        
        # self.gym.set_actor_root_state_tensor(self.sim, self.root_tensor)
        self.render(sync_frame_time=False)
        self.progress_buf += 1
        self.compute_observations()
        self.compute_tar_observations()

        self.time_out_buf = self.progress_buf > self.max_len_sample
        self.extras["time_outs"] = self.time_out_buf
        self.count_step += 1

        
        return self.get_quad_state(), self.get_tar_state()
    

    def reset(self, reset_buf=None):
        """ Reset all robots"""
        if reset_buf is None:
            reset_idx = torch.arange(self.num_envs, device=self.device)
        else:
            reset_idx = torch.nonzero(reset_buf).squeeze(-1)
            # if not len(reset_idx):
            #     print("################", reset_idx)
        # print("##### 12")
        if len(reset_idx):
            self.set_reset_idx(reset_idx)
            # print("##### 13")
            self.gym.set_actor_root_state_tensor(self.sim, self.root_tensor)
            # print("##### 14")
        self.gym.simulate(self.sim)
        self.render(sync_frame_time=False)
        return self.get_quad_state()

    def set_reset_idx(self, env_ids):
        num_resets = len(env_ids)
        self.count_step[env_ids] = 0

        # reset position
        # print(torch.tensor([self.tar_traj[idx, self.count_step[idx], :3] for idx in env_ids]).shape)
        # self.tar_root_states[env_ids, 0:3] = torch.tensor([self.tar_traj[idx, self.count_step[idx], :3] for idx in env_ids])
        # for idx in env_ids:
            # self.tar_root_states[idx, 0:3] = self.tar_traj[idx, self.count_step[idx], :3]
        # print(self.tar_traj.shape)
        # print(self.tar_traj.shape)
        self.tar_root_states[env_ids, 0] = 3
        self.tar_root_states[env_ids, 1] = 0
        # self.tar_root_states[env_ids, 0:3] = 0
        self.tar_root_states[env_ids, 2] = 7

        # reset linevels
        self.tar_root_states[env_ids, 7:10] = 0
        # self.tar_root_states[env_ids, 7:10] = self.tar_traj[env_ids, self.count_step[env_ids], 6:9]
        # reset angvels
        self.tar_root_states[env_ids, 10:13] = 0
        # reset quats
        self.tar_root_states[env_ids, 3:7] = 0
        self.tar_root_states[env_ids, 6] = 1.0

        self.tar_acc[env_ids] = rand_circle_point(num_resets, self.tar_acc_norm, self.device)
        # reset position
        # print(self.count_step.size(), self.tar_traj.size(),env_ids)
        # self.root_states[env_ids, 0:3] = self.tar_traj[env_ids, self.count_step[env_ids], :3]
        # self.root_states[env_ids, 0:2] = rand_circle_point(num_resets, 5, self.device)
        self.root_states[env_ids, 0:3] = 0
        self.root_states[env_ids, 2] = 7
        self.root_states[env_ids, :3] -= torch.tensor([0.21, 0, 0.05], device=self.device)
        # reset linevels
        self.root_states[env_ids, 7:10] = 0
        # reset angvels
        self.root_states[env_ids, 10:13] = 0
        # reset quats
        # print(self.tar_root_states[env_ids, :3] - self.root_states[env_ids, :3])
        # dir = self.tar_root_states[env_ids, :3] - self.root_states[env_ids, :3]
        # qua = self.direction_vectors_to_quaternion(env_ids, dir)
        # print(qua[0])
        # self.root_states[env_ids, 3:7].copy_(qua)
        self.root_states[env_ids, 3:7] = 0
        self.root_states[env_ids, 6] = 1
        # self.root_states[env_ids, 4] = 1
        # if torch.isnan(qua).any():
        #     # print(input_buffer[max(step+1-args.slide_size, 0):step+1])
        #     print("Nan detected in qua!!!")
        #     exit(0)

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.time_out_buf[env_ids] = 0


    def pre_physics_step(self, tar_state):
        # resets
        if self.counter % 250 == 0:
            print("self.counter:", self.counter)
        self.counter += 1

        
        tmp_tar_state = torch.zeros((self.num_envs, 13)).to(self.device)
        tmp_tar_state[:, :3] = tar_state[:, :3] - torch.tensor([0.21, 0, 0.05], device=self.device)
        tmp_tar_state[:, 3:7] = self.euler2qua(tar_state[:, 3:6])
        tmp_tar_state[:, 7:10] = tar_state[:, 6:9]
        tmp_tar_state[:, 10:13] = tar_state[:, 9:12]


        # -----------------------
        # $$$ This part is for moving target
        # inv_acc_idx = torch.nonzero((self.count_step + self.tar_acc_intervel) % (self.tar_acc_intervel * 2)).squeeze(-1)
        # change_acc_idx = torch.nonzero(self.count_step % (self.tar_acc_intervel * 2)).squeeze(-1)
        # self.tar_acc[inv_acc_idx] *= -1
        # self.tar_acc[change_acc_idx, :2] = rand_circle_point(len(change_acc_idx), self.tar_acc_norm, self.device)
        inv_acc_idx = torch.nonzero((self.count_step % self.tar_acc_intervel) == 0).squeeze(-1)
        self.tar_acc[inv_acc_idx] *= -1
        inv_acc_idx = torch.nonzero((self.count_step %( self.tar_acc_intervel * 2)) == 0).squeeze(-1)
        self.tar_acc[inv_acc_idx] *= -1
        # change_acc_idx = torch.nonzero(((self.count_step % (self.tar_acc_intervel * 4)) == 0)).squeeze(-1)
        # if len(change_acc_idx):
        #     self.tar_acc[change_acc_idx] = rand_circle_point(len(change_acc_idx), self.tar_acc_norm, self.device)

        # set position
        # self.tar_root_states[range(self.num_envs), 0:3] = 0
        self.tar_root_states[:, 2] = 7
        # set linearvels
        # print(f"Before Velocity in x: {self.tar_root_states[0, 7]}, Acceleration in x: {self.tar_acc[0, 0]}")
        # print(self.tar_root_states.shape, self.tar_acc.shape)
        self.tar_root_states[:, 7:9] += self.tar_acc * self.cfg.sim.dt
        # print(f"After Velocity in x: {self.tar_root_states[0, 7]}, Acceleration in x: {self.tar_acc[0, 0]}")
        self.tar_root_states[:, 9] = 0
        # set angvels
        self.tar_root_states[:, 10:13] = 0
        # set quats
        self.tar_root_states[:, 3:7] = 0
        self.tar_root_states[:, 6] = 1

        # # $$$ This part if for still target
        # self.tar_root_states[:, 0] = 3
        # self.tar_root_states[:, 1] = 0
        # # self.tar_root_states[env_ids, 0:3] = 0
        # self.tar_root_states[:, 2] = 7

        # # reset linevels
        # self.tar_root_states[:, 7:10] = 0
        # # self.tar_root_states[env_ids, 7:10] = self.tar_traj[env_ids, self.count_step[env_ids], 6:9]
        # # reset angvels
        # self.tar_root_states[:, 10:13] = 0
        # # reset quats
        # self.tar_root_states[:, 3:7] = 0
        # self.tar_root_states[:, 6] = 1.0

        # -----------------------

        self.root_states.copy_(tmp_tar_state)
        self.gym.set_actor_root_state_tensor(self.sim, self.root_tensor)


    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)

    def compute_observations(self):
        # print("obs_buf:", self.obs_buf.size(), self.obs_buf)
        # print("root_positions", self.root_positions.size(), self.root_positions)
        self.obs_buf[..., :3] = self.root_positions
        self.obs_buf[..., 3:7] = self.root_quats
        self.obs_buf[..., 7:10] = self.root_linvels
        self.obs_buf[..., 10:13] = self.root_angvels

        return self.obs_buf
    
    def compute_tar_observations(self):
        self.tar_obs_buf[..., :3] = self.tar_root_positions
        self.tar_obs_buf[..., 3:7] = self.tar_root_quats
        self.tar_obs_buf[..., 7:10] = self.tar_root_linvels
        self.tar_obs_buf[..., 10:13] = self.tar_root_angvels

        return self.tar_obs_buf

    def set_start_pos(self, robot_start_pos, tar_start_pos):
        self.initial_root_states[:, :3] = robot_start_pos
        self.initial_tar_states[:, :3] = tar_start_pos

    def qua2euler(self, qua):
        rotation_matrices = quaternion_to_matrix(
            qua[:, [3, 0, 1, 2]])
        euler_angles = matrix_to_euler_angles(
            rotation_matrices, "ZYX")[:, [2, 1, 0]]
        return euler_angles

    def euler2qua(self, euler):
        rotation_matrices = euler_angles_to_matrix(euler, "ZYX")
        qua = matrix_to_quaternion(rotation_matrices)[:, [3, 2, 1, 0]]
        return qua
    
    def get_camera_output(self):
        self.gym.start_access_image_tensors(self.sim)
        tmp_camera_root_tensors = torch.stack(self.camera_root_tensors)
        self.gym.end_access_image_tensors(self.sim)
        # print("!!!!!!camera device:", tmp_camera_root_tensors)
        return tmp_camera_root_tensors
    
    def get_camera_dep_output(self):
        self.gym.start_access_image_tensors(self.sim)
        tmp_camera_dep_root_tensors = torch.stack(self.camera_dep_root_tensors)
        self.gym.end_access_image_tensors(self.sim)
        # print(tmp_camera_dep_root_tensors.device)
        return tmp_camera_dep_root_tensors
    
    def get_camera_seg_output(self):
        self.gym.start_access_image_tensors(self.sim)
        tmp_camera_seg_root_tensors = torch.stack(self.camera_seg_root_tensors)
        self.gym.end_access_image_tensors(self.sim)
        # print(tmp_camera_dep_root_tensors.device)
        return tmp_camera_seg_root_tensors
    
    def get_camera_dep_seg_output(self):
        self.gym.start_access_image_tensors(self.sim)
        tmp_camera_dep_root_tensors = torch.stack(self.camera_dep_root_tensors)
        tmp_camera_seg_root_tensors = torch.stack(self.camera_seg_root_tensors)
        self.gym.end_access_image_tensors(self.sim)
        # print(tmp_camera_dep_root_tensors.device)
        return tmp_camera_dep_root_tensors, tmp_camera_seg_root_tensors

    def save_camera_output(self, file_name="tmp.png", file_path="/home/wangzimo/VTT/VTT/aerial_gym/scripts/camera_output/frames/", idx=0):
        filepath = file_path + file_name
        self.gym.write_camera_image_to_file(self.sim, self.envs[idx], self.camera_handles[idx], gymapi.IMAGE_COLOR, filepath)
        return self.gym.get_camera_image(self.sim, self.envs[idx], self.camera_handles[idx], gymapi.IMAGE_COLOR)
    
    def save_camera2_output(self, file_name="tmp.png", file_path="/home/wangzimo/VTT/VTT/aerial_gym/scripts/camera_output/frames/", idx=0):
        filepath = file_path + file_name
        self.gym.write_camera_image_to_file(self.sim, self.envs[idx], self.camera2_handles[idx], gymapi.IMAGE_COLOR, filepath)
        return self.gym.get_camera_image(self.sim, self.envs[idx], self.camera2_handles[idx], gymapi.IMAGE_COLOR)

    def get_quad_state(self):
        self.compute_observations()
        obs = self.obs_buf.clone()
        # new state generated by aerial gym
        new_state_ag = torch.zeros((self.num_envs, 12)).to(self.device)
        new_state_ag[:, :3] = obs[:, :3] + torch.tensor([0.21, 0, 0.05], device=self.device) # position
        new_state_ag[:, 3:6] = self.qua2euler(obs[:, 3:7]) # orientation
        # new_state_ag[3] = quat_axis(self.root_quats, 0)[0, 0] # orientation
        new_state_ag[:, 6:9] = obs[:, 7:10] # linear acceleration
        new_state_ag[:, 9:12] = obs[:, 10:13] # angular acceleration
        return new_state_ag.detach()

    def get_tar_state(self):
        self.compute_tar_observations()
        obs = self.tar_obs_buf.clone()
        new_state_tar = torch.zeros((self.num_envs, 12)).to(self.device)
        new_state_tar[:, :3] = obs[:, :3]
        new_state_tar[:, 3:6] = self.qua2euler(obs[:, 3:7]) # orientation
        new_state_tar[:, 6:9] = obs[:, 7:10] # linear acceleration
        new_state_tar[:, 9:12] = obs[:, 10:13] # angular acceleration
        return new_state_tar.detach()

    def check_out_sight(self):
        # """Must call compute_observations and compute_tar_observations before"""
        dep_image = self.get_camera_dep_output()
        sum_dep_image = torch.sum(dep_image, dim=(1, 2))
        out_sight = torch.where(sum_dep_image == 0, torch.tensor(0), torch.tensor(1)).squeeze(-1)
        return out_sight
    
    def check_timeout(self):
        return self.time_out_buf
    
    def check_out_space(self):
        ones = torch.ones_like(self.reset_buf)
        out_space = torch.zeros_like(self.reset_buf)
        obs = self.obs_buf.clone()
        
        out_space = torch.where(torch.logical_or(obs[:, 0] > 10, obs[:, 0] < -10), ones, out_space)
        out_space = torch.where(torch.logical_or(obs[:, 1] > 10, obs[:, 1] < -10), ones, out_space)
        out_space = torch.where(torch.logical_or(obs[:, 2] > 10, obs[:, 2] < 0), ones, out_space)
        return out_space
            
    def set_reset_to(self, tar_state):
        reset_idx = torch.arange(self.num_envs, device=self.device)
        tar_state = tar_state.detach()
        tmp_tar_state = torch.zeros((self.num_envs, 13)).to(self.device)
        tmp_tar_state[reset_idx, :3] = tar_state[reset_idx, :3]
        tmp_tar_state[reset_idx, 3:7] = self.euler2qua(tar_state[reset_idx, 3:6])
        tmp_tar_state[reset_idx, 7:10] = tar_state[reset_idx, 6:9]
        tmp_tar_state[reset_idx, 10:13] = tar_state[reset_idx, 9:12]
        
        self.root_states.copy_(tmp_tar_state)
        self.gym.set_actor_root_state_tensor(self.sim, self.root_tensor)
        
        
    def check_reset_out(self):
        # # $$$ First place
        # seg_image = self.get_camera_seg_output().to(device=self.device)
        # sum_seg_image = torch.sum(seg_image, dim=(1, 2))
        # # print(sum_dep_image)
        # # print(torch.tensor(1, device=self.device), torch.tensor(0, device=self.device))
        # out_sight = torch.where(sum_seg_image == 0, torch.tensor(1, device=self.device), torch.tensor(0, device=self.device)).squeeze(-1)
        # out_sight_idx = torch.nonzero(out_sight).squeeze(-1)
        # # if len(out_sight_idx):
        # #     print("out_sight:", out_sight_idx)
        # # print("##### 7")

        ones = torch.ones_like(self.reset_buf)
        out_space = torch.zeros_like(self.reset_buf)
        obs = self.obs_buf.clone()
        out_space = torch.where(torch.logical_or(obs[:, 0] > 15, obs[:, 0] < -15), ones, out_space)
        out_space = torch.where(torch.logical_or(obs[:, 1] > 15, obs[:, 1] < -15), ones, out_space)
        out_space = torch.where(torch.logical_or(obs[:, 2] > 15, obs[:, 2] < 0), ones, out_space)
        out_space = torch.where(torch.any(torch.isnan(obs[:, :3]), dim=1).bool(), ones, out_space)
        out_space_idx = torch.nonzero(out_space).squeeze(-1)
        # if len(out_space_idx):
        #     print("out_space:", out_space_idx)
        # print("##### 8")
        out_time = self.time_out_buf
        out_time_idx = torch.nonzero(out_time).squeeze(-1)
        # if len(out_time_idx):
        #     print("out_time:", out_time_idx)
        
        # $$$ Another first place
        # reset_buf = torch.logical_or(out_space, torch.logical_or(out_sight, out_time))
        reset_buf = torch.logical_or(out_space, out_time)
        reset_idx = torch.nonzero(reset_buf).squeeze(-1)
        
        
        return reset_buf, reset_idx

    def update_target_traj(self):
        
        self.tar_traj = next(self.tar_traj_iter)
        while self.tar_traj.size(0) < self.num_envs:
            print("Skip one data for limit number")
            self.tar_traj = next(self.tar_traj_iter)

    def get_relative_distance(self):
        quad_state = self.get_quad_state()
        tar_state = self.get_tar_state()
        quad_pos = quad_state[:, :3]
        tar_pos = tar_state[:, :3]
        relative_distance = tar_pos - quad_pos
        return relative_distance

    def get_future_relative_distance(self, len=50, sample_fre=10):
        quad_state = self.get_quad_state()
        quad_pos = quad_state[:, :3]
        
        future_tar_state = []
        for i in range(self.num_envs):
            future_tar_state.append(self.tar_traj[i, self.count_step[i]:self.count_step[i]+len:sample_fre, :3])
        # print(future_tar_state)
        future_tar_state = torch.stack(future_tar_state, dim=0)
        expanded_quad_pos = quad_pos.unsqueeze(1).expand(-1, len // sample_fre, -1)
        future_relative_distance = future_tar_state[:, :, :3] - expanded_quad_pos
        return future_relative_distance
    


    def direction_vectors_to_quaternion(self, env_ids, direction_vectors):


        init_vector = self.initial_vector[env_ids]
        # print(direction_vectors.shape, init_vector.shape)
        # 计算旋转轴
        rotation_axes = torch.cross(init_vector, direction_vectors, dim=-1)
        # print(direction_vectors.shape, init_vector.shape)
        # 计算夹角的余弦值
        cos_angles = F.cosine_similarity(direction_vectors, init_vector, dim=-1)
        if torch.isnan(cos_angles).any():
            # print(input_buffer[max(step+1-args.slide_size, 0):step+1])
            print("Nan detected in cos_angles!!!")
            exit(0)
        angles = torch.acos(cos_angles)

        # 处理方向向量与初始方向相同的情况
        valid_mask = cos_angles < 1.0  # 有效的夹角计算
        valid_angles = angles[valid_mask]
        # print(valid_angles.shape, valid_mask.shape)
        valid_rotation_axes = rotation_axes[valid_mask]

        # 归一化有效的旋转轴
        valid_rotation_axes /= torch.norm(valid_rotation_axes, dim=-1, keepdim=True)

        # 创建四元数
        quaternions = torch.zeros(direction_vectors.shape[0], 4, dtype=direction_vectors.dtype, device=direction_vectors.device)
        quaternions[valid_mask, 3] = torch.cos(valid_angles / 2)  # w
        quaternions[valid_mask, :3] = valid_rotation_axes * torch.sin(valid_angles / 2).unsqueeze(-1)  # (x, y, z)

        if torch.isnan(quaternions).any():
            # print(input_buffer[max(step+1-args.slide_size, 0):step+1])
            print("Nan detected in quaternions!!!")
            exit(0)
        # 返回四元数
        return quaternions
    
    def render(self, sync_frame_time=True):
        # # $$$ Second place
        # # print("##### 5.1")
        # # Fetch results
        # self.gym.fetch_results(self.sim, True) # use only when device is not "cpu"
        # # Step graphics. Skipping this causes the onboard robot camera tensors to not be updated
        # # print("##### 5.2")
        # self.gym.step_graphics(self.sim)
        # # print("##### 5.3")
        # self.gym.render_all_camera_sensors(self.sim)
        # # print("##### 5.4")
        # # if viewer exists update it based on requirement
        if self.viewer:
            # print("##### 5.5")
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # update viewer based on requirement
            if self.enable_viewer_sync:
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)