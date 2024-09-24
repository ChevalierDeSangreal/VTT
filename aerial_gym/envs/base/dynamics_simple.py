import sys
sys.path.append('/home/zim/Documents/python/VTT')
import numpy as np
import casadi as ca
import torch
import os
import json
from pathlib import Path

"""
On the base of dynamics_newton,
Change Controller to high level.
"""

class MyDynamics:

    def __init__(self, modified_params={}):
        """
        Initialzie quadrotor dynamics
        Args:
            modified_params (dict, optional): dynamic mismatch. Defaults to {}.
        """
        with open(
            os.path.join(Path(__file__).parent.absolute(), "config_quad_isaac.json"),
            "r"
        ) as infile:
            self.cfg = json.load(infile)

        # update with modified parameters
        self.cfg.update(modified_params)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
        self.device = device
        

        # NUMPY PARAMETERS
        self.mass = self.cfg["mass"]
        self.kinv_ang_vel_tau = np.array(self.cfg["kinv_ang_vel_tau"])

        # self.inertia_vector = (
        #     self.mass / 12.0 * self.arm_length**2 *
        #     np.array(self.cfg["frame_inertia"])
        # )
        self.inertia_vector = np.array(self.cfg["inertia"])
        # print("Inertia Vector:", self.inertia_vector)
        # TORCH PARAMETERS
        # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.copter_params = SimpleNamespace(**self.copter_params)
        self.torch_translational_drag = torch.tensor(
            self.cfg["translational_drag"]
        ).float().to(device)
        self.torch_gravity = torch.tensor(self.cfg["gravity"]).to(device)
        self.torch_rotational_drag = torch.tensor(self.cfg["rotational_drag"]
                                                  ).float().to(device)
        self.torch_inertia_vector = torch.from_numpy(self.inertia_vector
                                                     ).float().to(device)

        self.torch_inertia_J = torch.diag(self.torch_inertia_vector).to(device)
        self.torch_inertia_J_inv = torch.diag(1 / self.torch_inertia_vector).to(device)
        self.torch_kinv_vector = torch.tensor(self.kinv_ang_vel_tau).float().to(device)
        self.torch_kinv_ang_vel_tau = torch.diag(self.torch_kinv_vector).to(device)

        # CASADI PARAMETERS
        self.ca_inertia_vector = ca.SX(self.inertia_vector)
        self.ca_inertia_vector_inv = ca.SX(1 / self.inertia_vector)
        self.ca_kinv_ang_vel_tau = ca.SX(np.array(self.kinv_ang_vel_tau))

    @staticmethod
    def world_to_body_matrix(attitude):
        """
        Creates a transformation matrix for directions from world frame
        to body frame for a body with attitude given by `euler` Euler angles.
        :param euler: The Euler angles of the body frame.
        :return: The transformation matrix.
        """

        # check if we have a cached result already available
        roll = attitude[:, 0]
        pitch = attitude[:, 1]
        yaw = attitude[:, 2]

        Cy = torch.cos(yaw)
        Sy = torch.sin(yaw)
        Cp = torch.cos(pitch)
        Sp = torch.sin(pitch)
        Cr = torch.cos(roll)
        Sr = torch.sin(roll)

        # create matrix
        m1 = torch.transpose(torch.vstack([Cy * Cp, Sy * Cp, -Sp]), 0, 1)
        m2 = torch.transpose(
            torch.vstack(
                [Cy * Sp * Sr - Cr * Sy, Cr * Cy + Sr * Sy * Sp, Cp * Sr]
            ), 0, 1
        )
        m3 = torch.transpose(
            torch.vstack(
                [Cy * Sp * Cr + Sr * Sy, Cr * Sy * Sp - Cy * Sr, Cr * Cp]
            ), 0, 1
        )
        matrix = torch.stack((m1, m2, m3), dim=1)

        return matrix

    @staticmethod
    def to_euler_matrix(attitude):
        # attitude is [roll, pitch, yaw]
        pitch = attitude[:, 1]
        roll = attitude[:, 0]
        Cp = torch.cos(pitch)
        Sp = torch.sin(pitch)
        Cr = torch.cos(roll)
        Sr = torch.sin(roll)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        zero_vec_bs = torch.zeros(Sp.size()).to(device)
        ones_vec_bs = torch.ones(Sp.size()).to(device)

        # create matrix
        m1 = torch.transpose(
            torch.vstack([ones_vec_bs, zero_vec_bs, -Sp]), 0, 1
        )
        m2 = torch.transpose(torch.vstack([zero_vec_bs, Cr, Cp * Sr]), 0, 1)
        m3 = torch.transpose(torch.vstack([zero_vec_bs, -Sr, Cp * Cr]), 0, 1)
        matrix = torch.stack((m1, m2, m3), dim=1)

        # matrix = torch.tensor([[1, 0, -Sp], [0, Cr, Cp * Sr], [0, -Sr, Cp * Cr]])
        return matrix

    @staticmethod
    def euler_rate(attitude, angular_velocity):
        euler_matrix = MyDynamics.to_euler_matrix(attitude)
        together = torch.matmul(
            euler_matrix, torch.unsqueeze(angular_velocity.float(), 2)
        )
        # print("output euler rate", together.size())
        return torch.squeeze(together)

    @staticmethod
    def vector_to_euler_angles(vectors):
        vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True)
        
        pitch = torch.atan2(vectors[..., 1], torch.sqrt(vectors[..., 0]**2 + vectors[..., 2]**2))
        yaw = torch.atan2(vectors[..., 0], vectors[..., 2])
        roll = torch.zeros_like(pitch)
        
        return torch.stack([roll, pitch, yaw], dim=-1)
    
class SimpleDynamics(MyDynamics):

    def __init__(self, modified_params={}):
        super().__init__(modified_params=modified_params)
        torch.cuda.set_device(self.device)

    def body_acc_to_world(self, acc, attitude):

        
        world_to_body = self.world_to_body_matrix(attitude)
        body_to_world = torch.transpose(world_to_body, 1, 2)
        # print(acc.shape, body_to_world.shape)
        world_acc = torch.matmul(
            body_to_world, torch.unsqueeze(acc, 2)
        )
        
        return torch.squeeze(world_acc, 2)


    def _pretty_print(self, varname, torch_var):
        np.set_printoptions(suppress=1, precision=7)
        if len(torch_var) > 1:
            print("ERR: batch size larger 1", torch_var.size())
        print(varname, torch_var[0].detach().numpy())

    def __call__(self, state, action, dt):
        return self.simulate_quadrotor(action, state, dt)

    def simulate_quadrotor(self, action, state, dt):
        """
        Pytorch implementation of the dynamics in Flightmare simulator
        """
        # extract state
        position = state[:, :3]
        attitude = state[:, 3:6]
        velocity = state[:, 6:9]
        angular_velocity = state[:, 9:]


        acceleration = self.body_acc_to_world(action, attitude)
        # print(position.shape, acceleration.shape)
        position = (
            position * 0.9 + 0.1 * position.detach() + 0.5 * dt * dt * acceleration + dt * velocity
        )
        velocity = velocity * 0.9 + 0.1 * velocity.detach() + dt * acceleration

        # 2) angular acceleration

        attitude = self.vector_to_euler_angles(action)

        # set final state
        state = torch.hstack(
            (position, attitude, velocity, angular_velocity)
        )
        return state.float(), acceleration

class NRSimpleDynamics(MyDynamics):

    def __init__(self, modified_params={}):
        super().__init__(modified_params=modified_params)
        torch.cuda.set_device(self.device)

    def body_acc_to_world(self, acc, attitude):

        
        world_to_body = self.world_to_body_matrix(attitude)
        body_to_world = torch.transpose(world_to_body, 1, 2)
        # print(acc.shape, body_to_world.shape)
        world_acc = torch.matmul(
            body_to_world, torch.unsqueeze(acc, 2)
        )
        
        return torch.squeeze(world_acc, 2)


    def _pretty_print(self, varname, torch_var):
        np.set_printoptions(suppress=1, precision=7)
        if len(torch_var) > 1:
            print("ERR: batch size larger 1", torch_var.size())
        print(varname, torch_var[0].detach().numpy())

    def __call__(self, state, action, dt):
        return self.simulate_quadrotor(action, state, dt)

    def simulate_quadrotor(self, action, state, dt):
        """
        Pytorch implementation of the dynamics in Flightmare simulator
        """
        # extract state
        position = state[:, :3]
        attitude = state[:, 3:6]
        velocity = state[:, 6:9]
        angular_velocity = state[:, 9:]


        acceleration = self.body_acc_to_world(action, attitude)
        # print(position.shape, acceleration.shape)
        position = (
            position + 0.5 * dt * dt * acceleration + dt * velocity
        )
        velocity = velocity + dt * acceleration

        # 2) angular acceleration

        attitude = self.vector_to_euler_angles(action)

        # set final state
        state = torch.hstack(
            (position, attitude, velocity, angular_velocity)
        )
        return state.float(), acceleration



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    action = [-0.5395,  0.7426,  0.7635, -0.7364]
    action = torch.tensor([action, action]).to(device)

    state = [
        -0.203302, -8.12219, 0.484883, -0.15613, -0.446313, 0.25728, -4.70952,
        0.627684, -2.506545, -0.039999, -0.200001, 0.1
    ]
    state = torch.tensor([state, state]).to(device)
    # state = [2, 3, 4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    dyn = SimpleDynamics()
    new_state = dyn.simulate_quadrotor(
        action, state, 0.05
    )
    print("new state flightmare", new_state)

