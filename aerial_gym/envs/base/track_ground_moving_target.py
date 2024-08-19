# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
""" 
    在TrackGroundVer7的基础上
    使目标物体移动
    目标物体会沿着一个随机方向 以固定大小的速度进行移动
"""
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_euler_angles, euler_angles_to_matrix, matrix_to_quaternion

from aerial_gym import AERIAL_GYM_ROOT_DIR, AERIAL_GYM_ROOT_DIR

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import *
from aerial_gym.envs.base.base_task import BaseTask
from .track_ground_config import TrackGroundCfg

from aerial_gym.utils.helpers import asset_class_to_AssetOptions
from aerial_gym.utils.math import rand_circle_point

from aerial_gym.envs.base.dynamics_isaac import IsaacGymDynamics

class TrackGroundVer9(BaseTask):

    def __init__(self, cfg: TrackGroundCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg

        self.max_episode_length = int(self.cfg.env.episode_length_s / self.cfg.sim.dt)
        self.max_len_sample = self.cfg.env.max_sample_length
        self.debug_viz = False
        num_actors = 2

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device_id = sim_device
        self.headless = headless
        

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        # print("FFFFFFFFFFFUCKKKKKKKKKKKKKKKKKK YOUUUUUUUUUUUUU, ", self.sim_device_id, self.device)
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
        
        self.tar_dir = rand_circle_point(self.num_envs, 1, self.device)


    def create_sim(self):
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs()
        self.progress_buf = torch.zeros(
            self.cfg.env.num_envs, device=self.sim_device, dtype=torch.long)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.static_friction = 0
        plane_params.dynamic_friction = 0
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
        asset_options.override_inertia = True
        # asset_options.override_com = True
        tar_asset_options = asset_class_to_AssetOptions(self.cfg.tar_asset)

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options)

        tar_asset = self.gym.load_asset(
            self.sim, tar_asset_root, tar_asset_file)

        self.robot_num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.tar_num_bodies = self.gym.get_asset_rigid_body_count(tar_asset)
        # print("asset_file = ", asset_file)
        # print("asset_root = ", asset_root)
        # print("robot_num_bodies = ", self.robot_num_bodies)

        start_pose = gymapi.Transform()
        self.env_spacing = self.cfg.env.env_spacing
        env_lower = gymapi.Vec3(-self.env_spacing, -
                                self.env_spacing, -self.env_spacing)
        env_upper = gymapi.Vec3(
            self.env_spacing, self.env_spacing, self.env_spacing)
        self.actor_handles = []
        self.camera_handles = []
        self.envs = []
        self.camera_root_tensors = []
        self.camera_dep_root_tensors = []
        self.tar_seg_ids = []
        tar_seg_id_count = 114

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            
            pos = torch.tensor([0, 0, 3], device=self.device)
            start_pose.p = gymapi.Vec3(*pos)
            actor_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose, self.cfg.robot_asset.name, i, self.cfg.robot_asset.collision_mask, 0)
            self.robot_actor_index = self.gym.get_actor_index(env_handle, actor_handle, gymapi.IndexDomain.DOMAIN_ENV)
            self.robot_body_index = self.gym.get_actor_rigid_body_index(env_handle, actor_handle, 0, gymapi.IndexDomain.DOMAIN_ENV)
            self.robot_body_handle = self.gym.get_actor_rigid_body_handle(env_handle, actor_handle, 0)

            # print(f"In env {i}, robot_actor_index = {self.robot_actor_index}, robot_body_intex = {self.robot_body_index}, robot_body_handle = {self.robot_body_handle}")
            # print(f"robot_actor_handle = {actor_handle}")
            # print("env_handle = ", env_handle)

            # Create Cemara
            camera_properties = gymapi.CameraProperties()
            camera_properties.width = 224
            camera_properties.height = 224
            camera_properties.enable_tensors = True
            camera_handle = self.gym.create_camera_sensor(env_handle, camera_properties)
            camera_offset = gymapi.Vec3(0, 0, -0.2)
            camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(90))
            self.gym.attach_camera_to_body(camera_handle, env_handle, self.robot_body_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_POSITION)
            self.camera_handles.append(camera_handle)
            # camera_position = gymapi.Vec3(1.5, 1, 1.5)
            # camera_target = gymapi.Vec3(0, 0, 0)
            # self.gym.set_camera_location(camera_handle, env_handle, camera_position, camera_target)

            camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, camera_handle, gymapi.IMAGE_COLOR)
            torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)
            # print("!!!!!!!!!!!!!!!!", camera_tensor)
            self.camera_root_tensors.append(torch_camera_tensor)

            camera_dep_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, camera_handle, gymapi.IMAGE_SEGMENTATION)
            torch_camera_dep_tensor = gymtorch.wrap_tensor(camera_dep_tensor)
            self.camera_dep_root_tensors.append(torch_camera_dep_tensor)

            

            pos = torch.tensor([0, 0, 0.2], device=self.device)
            start_pose.p = gymapi.Vec3(*pos)
            tar_actor_handle = self.gym.create_actor(
                env_handle, tar_asset, start_pose, self.cfg.tar_asset.name, i, self.cfg.robot_asset.collision_mask, 0)
            self.tar_body_handle = self.gym.get_actor_rigid_body_handle(env_handle, tar_actor_handle, 0)
            self.tar_actor_index = self.gym.get_actor_index(env_handle, tar_actor_handle, gymapi.IndexDomain.DOMAIN_ENV)
            self.tar_body_index = self.gym.get_actor_rigid_body_index(env_handle, tar_actor_handle, 0, gymapi.IndexDomain.DOMAIN_ENV)
            # print(f"In env {i}, tar_actor_index = {self.tar_actor_index}, tar_body_intex = {self.tar_body_index}, tar_body_handle = {self.tar_body_handle}")
            # print(f"tar_actor_handle = {tar_actor_handle}")

            tar_seg_id_count += 1
            self.gym.set_rigid_body_segmentation_id(env_handle, tar_actor_handle, 0, tar_seg_id_count)
            # print("!!!!!!!!!!!!!!!!", self.gym.get_rigid_body_segmentation_id(env_handle, tar_actor_handle, self.tar_body_index))
            # Is this nesessary?
            # pos = torch.tensor([2, 0, 0], device=self.device)
            # wall_pose = gymapi.Transform()
            # wall_pose.p = gymapi.Vec3(*pos)
            self.robot_body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.actor_handles.append(tar_actor_handle)
        
        # self.camera_root_tensors = torch.stack(self.camera_root_tensors)

        self.robot_mass = 0
        for prop in self.robot_body_props:
            self.robot_mass += prop.mass
        #     # print(dir(prop.inertia))
        #     print("Inertia: ", prop.inertia.x, prop.inertia.y, prop.inertia.z, prop.mass, prop.com, prop.flags)
        # print("The actor handle list:", self.actor_handles)
        # print("The camera handle list:", self.camera_handles)
        print("Total robot mass: ", self.robot_mass)
        
        print("\n\n\n\n\n ENVIRONMENT CREATED \n\n\n\n\n\n")

    def step(self, actions):
        # print(actions)
        # step physics and render each frame
        for i in range(self.cfg.env.num_control_steps_per_env_step):
            self.pre_physics_step(actions)
            self.gym.simulate(self.sim)
            # NOTE: as per the isaacgym docs, self.gym.fetch_results must be called after self.gym.simulate, but not having it here seems to work fine
            # it is called in the render function.
            self.post_physics_step()
            
        # reset linevels
        # self.tar_root_states[env_ids, 7:10] = 0.2*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device)
        self.tar_root_states[:, 7:10] = 0
        self.tar_root_states[:, 7:9] = self.tar_dir
        # self.tar_root_states[env_ids, 9] = 0
        # reset angvels
        self.tar_root_states[:, 10:13] = 0
        # reset quats
        self.tar_root_states[:, 3:7] = 0
        self.tar_root_states[:, 6] = 2

        self.render(sync_frame_time=False)
        
        self.progress_buf += 1
        self.compute_observations()
        self.compute_tar_observations()

        # self.check_reset()
        # reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # if len(reset_env_ids) > 0:
        #     self.reset_idx(reset_env_ids)

        self.time_out_buf = self.progress_buf > self.max_len_sample
        self.extras["time_outs"] = self.time_out_buf

        # obs = self.obs_buf.clone()
        # # new state generated by aerial gym
        # new_state_ag = torch.zeros((self.num_envs, 12)).to(self.device)
        # new_state_ag[:, :3] = obs[:, :3] # position
        # new_state_ag[:, 3:6] = self.qua2euler(obs[:, 3:7]) # orientation
        # # new_state_ag[3] = quat_axis(self.root_quats, 0)[0, 0] # orientation
        # new_state_ag[:, 6:9] = obs[:, 7:10] # linear acceleration
        # new_state_ag[:, 9:12] = obs[:, 10:13] # angular acceleration


        # obs = self.tar_obs_buf.clone()
        # new_state_tar = torch.zeros((self.num_envs, 12)).to(self.device)
        # new_state_tar[:, :3] = obs[:, :3]
        # new_state_tar[:, 3:6] = self.qua2euler(obs[:, 3:7]) # orientation
        # new_state_tar[:, 6:9] = obs[:, 7:10] # linear acceleration
        # new_state_tar[:, 9:12] = obs[:, 10:13] # angular acceleration

        return self.get_quad_state(), self.get_tar_state()
    

    def reset(self, reset_buf=None, reset_quad_state=None):
        """ Reset all robots"""
        if reset_buf is None:
            # print("!!!!!!!!!!!!!!!!!!!!!!!")
            reset_idx = torch.arange(self.num_envs, device=self.device)
        else:
            reset_idx = torch.nonzero(reset_buf).squeeze(-1)
        if reset_quad_state is not None:
            self.set_reset_to(reset_quad_state)
        # print(reset_buf)
        if len(reset_idx):
            self.set_reset_idx(reset_idx)
        # self.set_reset_idx(torch.arange(self.num_envs, device=self.device))
        # print(self.root_states)
        if len(reset_idx) or (reset_quad_state is not None):
            self.gym.set_actor_root_state_tensor(self.sim, self.root_tensor)
        
        return self.get_quad_state()

    def set_reset_idx(self, env_ids):
        num_resets = len(env_ids)

        self.root_states[env_ids] = self.initial_root_states[env_ids]
        # reset position
        # self.root_states[env_ids, 0:3] = 3.0 * torch_rand_float(0, 1.0, (num_resets, 3), self.device)
        self.root_states[env_ids, 0:2] = rand_circle_point(num_resets, 5, self.device)
        # self.root_states[env_ids, 0:2] = 3.0 * torch_rand_float(-1.0, 1.0, (num_resets, 2), self.device)
        # self.root_states[env_ids, 0:2] = 0
        self.root_states[env_ids, 2] = 7
        # reset linevels
        # self.root_states[env_ids, 7:10] = 0.2*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device)
        self.root_states[env_ids, 7:10] = 0
        # reset angvels
        # self.root_states[env_ids, 10:13] = 0.2*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device)
        self.root_states[env_ids, 10:13] = 0
        # reset quats
        self.root_states[env_ids, 3:7] = 0
        self.root_states[env_ids, 6] = 1.0
        

        self.tar_root_states[env_ids] = self.initial_tar_states[env_ids]
        # reset position
        # self.tar_root_states[env_ids, 0:3] = 5.0 * torch_rand_float(0, 1.0, (num_resets, 3), self.device)
        # self.tar_root_states[env_ids, 2] = 0
        self.tar_root_states[env_ids, 0:3] = 0

        # tmp_num = int(num_resets / 2)
        # tmp_list = torch.randperm(num_resets)[:tmp_num]
        # self.tar_root_states[tmp_list, 0] *= -1
        # tmp_list = torch.randperm(num_resets)[:tmp_num]
        # self.tar_root_states[tmp_list, 1] *= -1
        # self.tar_root_states[env_ids, 0:3] = torch.randint(-3, 4, (num_resets, 3), dtype=torch.float, device=self.device)
        self.tar_root_states[env_ids, 2] = 0.2

        # reset linevels
        # self.tar_root_states[env_ids, 7:10] = 0.2*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device)
        self.tar_root_states[env_ids, 7:10] = 0
        self.tar_root_states[env_ids, 7:9] = self.tar_dir
        # self.tar_root_states[env_ids, 9] = 0
        # reset angvels
        self.tar_root_states[env_ids, 10:13] = 0
        # reset quats
        self.tar_root_states[env_ids, 3:7] = 0
        self.tar_root_states[env_ids, 6] = 1.0

        # self.gym.set_actor_root_state_tensor(self.sim, self.root_tensor)
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.time_out_buf[env_ids] = 0


    def pre_physics_step(self, _actions):
        # resets
        if self.counter % 250 == 0:
            print("self.counter:", self.counter)
        self.counter += 1

        
        actions = _actions.to(self.device)
        actions = tensor_clamp(
            actions, self.action_lower_limits, self.action_upper_limits)
        self.action_input[:] = actions

        # clear actions for reset envs
        self.forces[:] = 0.0
        self.torques[:, :] = 0.0

        # output_thrusts_mass_normalized, output_torques_inertia_normalized = self.controller(self.root_states, self.action_input)
        force_torques = self.dynamics.control_quadrotor(self.action_input, self.get_quad_state())
        output_thrusts_mass_normalized = force_torques[:, 0]
        output_torques_inertia_normalized = force_torques[:, 1:]
        # self.forces[:, self.robot_body_index, 2] = self.robot_mass * (-self.sim_params.gravity.z) * output_thrusts_mass_normalized
        self.forces[:, self.robot_body_index, 2] = output_thrusts_mass_normalized
        self.torques[:, self.robot_body_index] = output_torques_inertia_normalized
        # @@@
        # self.forces = torch.where(self.forces < 0, torch.zeros_like(self.forces), self.forces)
        # print("Forces:", self.forces[:, 0])
        # print("Torques:", self.torques[:, 0])
        # apply actions
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(
            self.forces), gymtorch.unwrap_tensor(self.torques), gymapi.LOCAL_SPACE)
        


    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)

    def compute_observations(self):
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
        return tmp_camera_root_tensors
    
    def get_camera_dep_output(self):
        self.gym.start_access_image_tensors(self.sim)
        tmp_camera_dep_root_tensors = torch.stack(self.camera_dep_root_tensors)
        self.gym.end_access_image_tensors(self.sim)
        return tmp_camera_dep_root_tensors

    def save_camera_output(self, file_name="tmp.png", file_path="/home/lab929/wzm/FYP/AGAPG/aerial_gym/scripts/camera_output/"):
        filepath = file_path + file_name
        self.gym.write_camera_image_to_file(self.sim, self.envs[1], self.camera_handles[1], gymapi.IMAGE_COLOR, filepath)
        return self.gym.get_camera_image(self.sim, self.envs[1], self.camera_handles[1], gymapi.IMAGE_COLOR)
    
    
    def get_quad_state(self):
        self.compute_observations()
        obs = self.obs_buf.clone()
        # new state generated by aerial gym
        new_state_ag = torch.zeros((self.num_envs, 12)).to(self.device)
        new_state_ag[:, :3] = obs[:, :3] # position
        new_state_ag[:, 3:6] = self.qua2euler(obs[:, 3:7]) # orientation
        # new_state_ag[3] = quat_axis(self.root_quats, 0)[0, 0] # orientation
        new_state_ag[:, 6:9] = obs[:, 7:10] # linear acceleration
        new_state_ag[:, 9:12] = obs[:, 10:13] # angular acceleration
        return new_state_ag

    def get_tar_state(self):
        self.compute_tar_observations()
        obs = self.tar_obs_buf.clone()
        new_state_tar = torch.zeros((self.num_envs, 12)).to(self.device)
        new_state_tar[:, :3] = obs[:, :3]
        new_state_tar[:, 3:6] = self.qua2euler(obs[:, 3:7]) # orientation
        new_state_tar[:, 6:9] = obs[:, 7:10] # linear acceleration
        new_state_tar[:, 9:12] = obs[:, 10:13] # angular acceleration
        return new_state_tar

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

        # self.gym.set_actor_root_state_tensor(self.sim, self.root_tensor)
        
    def reset_to(self, tar_state):
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
        dep_image = self.get_camera_dep_output()
        # print(dep_image.shape)
        sum_dep_image = torch.sum(dep_image, dim=(1, 2))
        # print("sum_dep_image:", sum_dep_image)
        out_sight = torch.where(sum_dep_image == 0, torch.tensor(1, device=self.device), torch.tensor(0, device=self.device)).squeeze(-1)
        out_sight_idx = torch.nonzero(out_sight).squeeze(-1)
        # print(len(out_sight))
        if len(out_sight_idx):
            print("out_sight:", out_sight_idx)
        
        ones = torch.ones_like(self.reset_buf)
        out_space = torch.zeros_like(self.reset_buf)
        obs = self.obs_buf.clone()
        out_space = torch.where(torch.logical_or(obs[:, 0] > 10, obs[:, 0] < -10), ones, out_space)
        out_space = torch.where(torch.logical_or(obs[:, 1] > 10, obs[:, 1] < -10), ones, out_space)
        out_space = torch.where(torch.logical_or(obs[:, 2] > 10, obs[:, 2] < 0), ones, out_space)
        out_space = torch.where(torch.any(torch.isnan(obs[:, :3]), dim=1).bool(), ones, out_space)
        out_space_idx = torch.nonzero(out_space).squeeze(-1)
        if len(out_space_idx):
            print("out_space:", out_space_idx)
        
        out_time = self.time_out_buf
        out_time_idx = torch.nonzero(out_time).squeeze(-1)
        if len(out_time_idx):
            print("out_time:", out_time_idx)
        
        reset_buf = torch.logical_or(out_space, torch.logical_or(out_sight, out_time))
        # reset_buf = torch.logical_or(out_space, out_time)
        reset_idx = torch.nonzero(reset_buf).squeeze(-1)
        
        
        return reset_buf, reset_idx
