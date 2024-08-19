import sys
sys.path.append('/home/cgv841/wzm/FYP/AGAPG')
import numpy as np
import casadi as ca
import torch
import os
import json
from pathlib import Path

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
        self.arm_length = self.cfg["arm_length"]
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


class IsaacGymDynamics(MyDynamics):

    def __init__(self, modified_params={}):
        super().__init__(modified_params=modified_params)
        torch.cuda.set_device(self.device)

    def linear_dynamics(self, force, attitude, velocity):
        """
        linear dynamics
        no drag so far
        """

        world_to_body = self.world_to_body_matrix(attitude)
        body_to_world = torch.transpose(world_to_body, 1, 2)

        # print("force in ld ", force.size())
        thrust = 1 / self.mass * torch.matmul(
            body_to_world, torch.unsqueeze(force, 2)
        )
        # print("thrust", thrust.size())
        # drag = velocity * TODO: dynamics.drag_coeff??
        thrust_min_grav = (
            thrust[:, :, 0] + self.torch_gravity +
            self.torch_translational_drag
        )
        return thrust_min_grav  # - drag

    def run_flight_control(self, thrust, av, body_rates, cross_prod):
        """
        thrust: command first signal (around 9.81)
        omega = av: current angular velocity
        command = body_rates: body rates in command
        """
        force = torch.unsqueeze(thrust, 1)

        # constants
        omega_change = torch.unsqueeze(body_rates - av, 2)
        kinv_times_change = torch.matmul(
            self.torch_kinv_ang_vel_tau.to(self.device), omega_change.to(self.device)
        )
        first_part = torch.matmul(self.torch_inertia_J.to(self.device), kinv_times_change.to(self.device))
        # print("first_part", first_part.size())
        body_torque_des = (
            first_part[:, :, 0] + cross_prod + self.torch_rotational_drag
        )
        # print(force.shape, body_torque_des.shape)
        thrust_and_torque = torch.unsqueeze(
            torch.cat((force, body_torque_des), dim=1), 2
        )
        return thrust_and_torque[:, :, 0]

    def _pretty_print(self, varname, torch_var):
        np.set_printoptions(suppress=1, precision=7)
        if len(torch_var) > 1:
            print("ERR: batch size larger 1", torch_var.size())
        print(varname, torch_var[0].detach().numpy())

    def __call__(self, state, action, dt):
        return self.simulate_quadrotor(action, state, dt)
    
    def control_quadrotor(self, action, state):
        # extract state
        position = state[:, :3]
        attitude = state[:, 3:6]
        velocity = state[:, 6:9]
        angular_velocity = state[:, 9:]

        # action is normalized between -1 and 1 --> rescale
        total_thrust = action[:, 0] * 7.5 + self.mass * (-self.torch_gravity[2])
        body_rates = action[:, 1:] * .5

        # ctl_dt ist simulation time,
        # remainer wird immer -sim_dt gemacht in jedem loop
        # precompute cross product (batch, 3, 1)
        # print(angular_velocity.shape)
        inertia_av = torch.matmul(
            self.torch_inertia_J.to(self.device), torch.unsqueeze(angular_velocity, 2)
        )[:, :, 0]
        cross_prod = torch.cross(angular_velocity, inertia_av, dim=1)

        force_torques = self.run_flight_control(
            total_thrust, angular_velocity, body_rates, cross_prod
        ).to(self.device)
        return force_torques
        

    def simulate_quadrotor(self, action, state, dt):
        """
        Pytorch implementation of the dynamics in Flightmare simulator
        """
        # extract state
        position = state[:, :3]
        attitude = state[:, 3:6]
        velocity = state[:, 6:9]
        angular_velocity = state[:, 9:]

        # action is normalized between -1 and 1 --> rescale
        total_thrust = action[:, 0] * 7.5 + self.mass * (-self.torch_gravity[2])
        body_rates = action[:, 1:] * .5

        # ctl_dt ist simulation time,
        # remainer wird immer -sim_dt gemacht in jedem loop
        # precompute cross product (batch, 3, 1)
        inertia_av = torch.matmul(
            self.torch_inertia_J.to(self.device), torch.unsqueeze(angular_velocity, 2)
        )[:, :, 0]
        cross_prod = torch.cross(angular_velocity, inertia_av, dim=1)

        force_torques = self.run_flight_control(
            total_thrust, angular_velocity, body_rates, cross_prod
        ).to(self.device)

        # 1) linear dynamics
        force_expanded = torch.unsqueeze(force_torques[:, 0], 1)
        f_s = force_expanded.size()
        force = torch.cat(
            (torch.zeros(f_s).to(self.device), torch.zeros(f_s).to(self.device), force_expanded), dim=1
        )

        acceleration = self.linear_dynamics(force, attitude, velocity)

        position = (
            position + 0.5 * dt * dt * acceleration + 0.5 * dt * velocity
        )
        velocity = velocity + dt * acceleration

        # 2) angular acceleration
        tau = force_torques[:, 1:]
        torch_inertia_J_inv = torch.inverse(self.torch_inertia_J.to(self.device))
        angular_acc = torch.matmul(
            torch_inertia_J_inv.to(self.device), torch.unsqueeze((tau - cross_prod).to(self.device), 2)
        )[:, :, 0]
        new_angular_velocity = angular_velocity + dt * angular_acc

        # other option: use quaternion
        # --> also slight error to flightmare, even when using euler, no idea why
        # from neural_control.trajectory.q_funcs import (
        #     q_dot_new, euler_to_quaternion, quaternion_to_euler
        # )
        # quaternion = euler_to_quaternion(
        #     attitude[0, 0].item(), attitude[0, 1].item(), attitude[0, 2].item()
        # )
        # print("quaternion", quaternion)
        # np.set_printoptions(suppress=1, precision=7)
        # av_test = angular_velocity[0].numpy()
        # quaternion_omega = np.array([av_test[0], av_test[1], av_test[2]])
        # print("quaternion_omega", quaternion_omega)
        # q_dot = q_dot_new(quaternion, quaternion_omega)
        # print("q dot", q_dot)
        # # integrate
        # new_quaternion = quaternion + dt * q_dot
        # print("new_quaternion", new_quaternion)
        # new_quaternion = new_quaternion / np.linalg.norm(new_quaternion)
        # print("new_quaternion", new_quaternion)
        # new_euler = quaternion_to_euler(new_quaternion)
        # print("new euler", new_euler)

        # pretty_print("attitude before", attitude)

        attitude = attitude + dt * self.euler_rate(attitude, angular_velocity)

        # set final state
        state = torch.hstack(
            (position, attitude, velocity, new_angular_velocity)
        )
        return state.float()





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
    dyn = IsaacGymDynamics()
    new_state = dyn.simulate_quadrotor(
        action, state, 0.05
    )
    action = dyn.control_quadrotor(action, state)
    print("action", action)
    print("new state flightmare", new_state)

