import argparse
import time
import os
import json
import sys
sys.path.append('/home/wangzimo/VTT/VTT')

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared
from tqdm import tqdm

from aerial_gym.utils.q_funcs import (
    q_dot_q,
    quaternion_inverse,
    quaternion_to_euler,
)

# from utils.visualization import debug_plot, draw_poly
"""
Autor: Elia Kaufmann
Script for generating random and geometric quadrotor trajectories
"""

from aerial_gym.envs.base.dynamics_isaac import MyDynamics


class Quad(MyDynamics):
    def __init__(self, max_thrust_per_motor):
        """
        :param mass: mass of the quadrotor in [kg]
        :param max_thrust_per_motor: maximum thrust in [N] per motor
        """
        super().__init__()
        self.max_thrust_per_motor = max_thrust_per_motor

        self.J = self.inertia_vector
        h = self.arm_length / np.sqrt(2.0)
        self.x_f = np.array([h, -h, -h, h])
        self.y_f = np.array([-h, -h, h, h])

        # For z thrust torque calculation
        self.c = 0.013  # m   (z torque generated by each motor)
        self.z_l_tau = np.array([-self.c, self.c, -self.c, self.c])




def smooth(x, window_len=11, window="hanning"):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd int
        window: the type of window from 'flat', 'hanning', 'hamming',
            'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
    numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array
    instead of a string
    NOTE: length(output) != length(input), to correct this: return y
    [(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett','blackman'"
        )

    # s = np.r_[x[(window_len - 1) // 2:0:-1], x, x[-2:-(window_len - 1) // 2:-1]]

    x_start = np.repeat(x[0], (window_len - 1) // 2)
    x_end = np.repeat(x[-1], (window_len - 1) // 2)
    s = np.concatenate([x_start, x, x_end])

    if window == "flat":  # moving average
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")

    y = np.convolve(w / w.sum(), s, mode="valid")
    return y


def compute_full_traj(quad, t_np, pos_np, vel_np, alin_np):
    len_traj = t_np.shape[0]
    dt = np.mean(np.diff(t_np))

    # Add gravity to accelerations
    gravity = 9.81
    thrust_np = alin_np + np.tile(np.array([[0, 0, 1]]), (len_traj, 1)) * gravity
    # Compute body axes
    z_b = thrust_np / np.sqrt(np.sum(thrust_np**2, 1))[:, np.newaxis]
    # new way to compute attitude:
    # https://math.stackexchange.com/questions/2251214/calculate-quaternions-from-two-directional-vectors
    e_z = np.array([[0.0, 0.0, 1.0]])
    q_w = 1.0 + np.sum(e_z * z_b, axis=1)
    q_xyz = np.cross(e_z, z_b)
    att_np = 0.5 * np.concatenate([np.expand_dims(q_w, axis=1), q_xyz], axis=1)
    att_np = att_np / np.sqrt(np.sum(att_np**2, 1))[:, np.newaxis]

    rate_np = np.zeros_like(pos_np)
    f_t = np.zeros((len_traj, 1))

    # Use numerical differentiation of quaternions
    q_dot = np.gradient(att_np, axis=0) / dt
    w_int = np.zeros((len_traj, 3))
    for i in range(len_traj):
        w_int[i, :] = 2.0 * q_dot_q(quaternion_inverse(att_np[i, :]), q_dot[i])[1:]
        f_t[i, 0] = quad.mass * z_b[i].dot(thrust_np[i, :].T)
    rate_np[:, 0] = w_int[:, 0]
    rate_np[:, 1] = w_int[:, 1]
    rate_np[:, 2] = w_int[:, 2]

    minimize_yaw_rate = True
    n_iter_yaw_fix = 20
    if minimize_yaw_rate:
        for iter_yaw_fix in range(n_iter_yaw_fix):
            # print(
            #     "Maximum yawrate before adaption %d / %d: %.6f" %
            #     (iter_yaw_fix, n_iter_yaw_fix, np.max(np.abs(rate_np[:, 2])))
            # )
            q_new = att_np
            yaw_corr_acc = 0.0
            for i in range(1, len_traj):
                yaw_corr = -rate_np[i, 2] * dt
                yaw_corr_acc += yaw_corr
                q_corr = np.array(
                    [np.cos(yaw_corr_acc / 2.0), 0.0, 0.0, np.sin(yaw_corr_acc / 2.0)]
                )
                q_new[i, :] = q_dot_q(att_np[i, :], q_corr)
                w_int[i, :] = (
                    2.0 * q_dot_q(quaternion_inverse(att_np[i, :]), q_dot[i])[1:]
                )

            q_new_dot = np.gradient(q_new, axis=0) / dt
            for i in range(1, len_traj):
                w_int[i, :] = (
                    2.0 * q_dot_q(quaternion_inverse(q_new[i, :]), q_new_dot[i])[1:]
                )

            att_np = q_new
            rate_np[:, 0] = w_int[:, 0]
            rate_np[:, 1] = w_int[:, 1]
            rate_np[:, 2] = w_int[:, 2]
            # print(
            #     "Maximum yawrate after adaption: %.3f" %
            #     np.max(np.abs(rate_np[:, 2]))
            # )
            if np.max(np.abs(rate_np[:, 2])) < 0.005:
                break

    arot_np = np.gradient(rate_np, axis=0)
    trajectory = np.concatenate(
        [pos_np, att_np, vel_np, rate_np, alin_np, arot_np], axis=1
    )
    motor_inputs = np.zeros((pos_np.shape[0], 4))

    # Compute inputs
    rate_dot = np.gradient(rate_np, axis=0) / dt
    rate_x_Jrate = np.array(
        [
            (quad.J[2] - quad.J[1]) * rate_np[:, 2] * rate_np[:, 1],
            (quad.J[0] - quad.J[2]) * rate_np[:, 0] * rate_np[:, 2],
            (quad.J[1] - quad.J[0]) * rate_np[:, 1] * rate_np[:, 0],
        ]
    ).T

    tau = rate_dot * quad.J[np.newaxis, :] + rate_x_Jrate
    b = np.concatenate((tau, f_t), axis=-1)
    a_mat = np.concatenate(
        (
            quad.y_f[np.newaxis, :],
            -quad.x_f[np.newaxis, :],
            quad.z_l_tau[np.newaxis, :],
            np.ones_like(quad.z_l_tau)[np.newaxis, :],
        ),
        0,
    )

    # for i in range(len_traj):
    #     motor_inputs[i, :] = np.linalg.solve(a_mat, b[i, :])

    return trajectory, np.zeros(4), t_np


def compute_random_trajectory(
    quad,
    arena_bound_max,
    arena_bound_min,
    freq_x,
    freq_y,
    freq_z,
    duration=30.0,
    dt=0.01,
    seed=0,
):
    # print("Computing random trajectory!")
    # assert dt == 0.01

    debug = False

    # kernel to map functions that repeat exactly
    # print("seed is: %d" % seed)
    kernel_y = (
        ExpSineSquared(length_scale=freq_x, periodicity=17)
        + ExpSineSquared(length_scale=3.0, periodicity=23)
        + ExpSineSquared(length_scale=4.0, periodicity=51)
    )
    kernel_x = (
        ExpSineSquared(length_scale=freq_y, periodicity=37)
        + ExpSineSquared(length_scale=3.0, periodicity=61)
        + ExpSineSquared(length_scale=4.0, periodicity=13)
    )
    kernel_z = (
        ExpSineSquared(length_scale=freq_z, periodicity=19)
        + ExpSineSquared(length_scale=3.0, periodicity=29)
        + ExpSineSquared(length_scale=4.0, periodicity=53)
    )

    gp_x = GaussianProcessRegressor(kernel=kernel_x)
    gp_y = GaussianProcessRegressor(kernel=kernel_y)
    gp_z = GaussianProcessRegressor(kernel=kernel_z)

    t_coarse = np.linspace(0.0, duration, int(duration / 0.1), endpoint=False)
    t_vec, dt = np.linspace(
        0.0, duration, int(duration / dt), endpoint=False, retstep=True
    )

    t = cs.MX.sym("t")
    # t_speed is a function starting at zero and ending at zero that
    # modulates time
    # casadi cannot do symbolic integration --> write down the integrand by
    # hand of 2.0*sin^2(t)
    # t_adj = 2.0 * (t / 2.0 - cs.sin(2.0 / duration * cs.pi * t) /
    # (4.0 * cs.pi / duration))
    tau = t / duration
    t_adj = (
        1.524
        * duration
        * (
            -(
                8 * cs.cos(tau * cs.pi) * cs.constpow(cs.sin(tau * cs.pi), 5)
                + 10 * cs.cos(tau * cs.pi) * cs.constpow(cs.sin(tau * cs.pi), 3)
                + 39 * cs.sin(tau * cs.pi) * cs.cos(tau * cs.pi)
                + 12 * cs.sin(2 * tau * cs.pi) * cs.cos(2 * tau * cs.pi)
                - 63 * tau * cs.pi
            )
            / (96 * cs.pi)
        )
    )

    f_t_adj = cs.Function("t_adj", [t], [t_adj])
    scaled_time = f_t_adj(t_vec)

    # print("sampling x...")
    x_sample_hr = gp_x.sample_y(t_coarse[:, np.newaxis], 1, random_state=seed)
    # print("sampling y...")
    y_sample_hr = gp_y.sample_y(t_coarse[:, np.newaxis], 1, random_state=seed + 1)
    # print("sampling z...")
    z_sample_hr = gp_z.sample_y(t_coarse[:, np.newaxis], 1, random_state=seed + 2)

    pos_np = np.concatenate([x_sample_hr, y_sample_hr, z_sample_hr], axis=1)
    # scale to arena bounds
    max_traj = np.max(pos_np, axis=0)
    min_traj = np.min(pos_np, axis=0)
    pos_centered = pos_np - (max_traj + min_traj) / 2.0
    pos_scaled = (
        pos_centered * (arena_bound_max - arena_bound_min) / (max_traj - min_traj)
    )
    pos_arena = pos_scaled + (arena_bound_max + arena_bound_min) / 2.0

    if debug:
        plt.plot(pos_arena[:, 0], label="x")
        plt.plot(pos_arena[:, 1], label="y")
        plt.plot(pos_arena[:, 2], label="z")
        plt.legend()
        plt.show()

    # rescale time to get smooth start and end states
    pos_blub_x = interpolate.interp1d(
        t_coarse, pos_arena[:, 0], kind="cubic", fill_value="extrapolate"
    )
    pos_blub_y = interpolate.interp1d(
        t_coarse, pos_arena[:, 1], kind="cubic", fill_value="extrapolate"
    )
    pos_blub_z = interpolate.interp1d(
        t_coarse, pos_arena[:, 2], kind="cubic", fill_value="extrapolate"
    )
    pos_arena = np.concatenate(
        [pos_blub_x(scaled_time), pos_blub_y(scaled_time), pos_blub_z(scaled_time)],
        axis=1,
    )

    pos_arena = np.concatenate(
        [
            smooth(np.squeeze(pos_arena[:, 0]), window_len=11)[:, np.newaxis],
            smooth(np.squeeze(pos_arena[:, 1]), window_len=11)[:, np.newaxis],
            smooth(np.squeeze(pos_arena[:, 2]), window_len=11)[:, np.newaxis],
        ],
        axis=1,
    )

    # compute numeric derivative & smooth things
    vel_arena = np.gradient(pos_arena, axis=0) / dt
    vel_arena = np.concatenate(
        [
            smooth(np.squeeze(vel_arena[:, 0]), window_len=11)[:, np.newaxis],
            smooth(np.squeeze(vel_arena[:, 1]), window_len=11)[:, np.newaxis],
            smooth(np.squeeze(vel_arena[:, 2]), window_len=11)[:, np.newaxis],
        ],
        axis=1,
    )
    acc_arena = np.gradient(vel_arena, axis=0) / dt
    acc_arena = np.concatenate(
        [
            smooth(np.squeeze(acc_arena[:, 0]), window_len=11)[:, np.newaxis],
            smooth(np.squeeze(acc_arena[:, 1]), window_len=11)[:, np.newaxis],
            smooth(np.squeeze(acc_arena[:, 2]), window_len=11)[:, np.newaxis],
        ],
        axis=1,
    )
    t_np = t_vec

    trajectory, motor_inputs, t_vec = compute_full_traj(
        quad, t_np, pos_arena, vel_arena, acc_arena
    )

    return trajectory, motor_inputs, t_vec


def compute_geometric_trajectory(quad, duration=30.0, dt=0.001):
    print("Computing geometric trajectory!")
    assert dt == 0.001

    debug = False

    # define position trajectory symbolically
    t = cs.MX.sym("t")
    # t_speed is a function starting at zero and ending at zero that
    # modulates time
    # casadi cannot do symbolic integration --> write down the integrand by
    # hand of 2.0*sin^2(t)
    # t_adj = 2.0 * (t / 2.0 - cs.sin(2.0 / duration * cs.pi * t) /
    # (4.0 * cs.pi / duration))
    tau = t / duration
    t_adj = (
        1.524
        * duration
        * (
            -(
                8 * cs.cos(tau * cs.pi) * cs.constpow(cs.sin(tau * cs.pi), 5)
                + 10 * cs.cos(tau * cs.pi) * cs.constpow(cs.sin(tau * cs.pi), 3)
                + 39 * cs.sin(tau * cs.pi) * cs.cos(tau * cs.pi)
                + 12 * cs.sin(2 * tau * cs.pi) * cs.cos(2 * tau * cs.pi)
                - 63 * tau * cs.pi
            )
            / (96 * cs.pi)
        )
    )

    # sphere trajectory rotating around x-axis
    radius_x = 5.0
    radius_y = 3.5
    radius_z = 2.5

    # fast config
    # freq_slow = 0.009
    # freq_fast = 0.33
    # slow config
    freq_slow = 0.02
    freq_fast = 0.12
    pos_x = 3.0 + radius_x * (
        cs.sin(2.0 * cs.pi * freq_fast * t_adj)
        * cs.cos(2.0 * cs.pi * freq_slow * t_adj)
    )
    pos_y = 1.0 + radius_y * (cs.cos(2.0 * cs.pi * freq_fast * t_adj))
    pos_z = 3.5 + radius_z * (
        cs.sin(2.0 * cs.pi * freq_fast * t_adj)
        * cs.sin(2.0 * cs.pi * freq_slow * t_adj)
    )

    # TODO: define yaw trajectory
    pos = cs.vertcat(pos_x, pos_y, pos_z)
    vel = cs.jacobian(pos, t)
    acc = cs.jacobian(vel, t)
    jerk = cs.jacobian(acc, t)
    snap = cs.jacobian(jerk, t)

    t_vec, dt = np.linspace(
        0.0, duration, int(duration / dt), endpoint=False, retstep=True
    )

    f_t_adj = cs.Function("t_adj", [t], [t_adj])
    f_pos = cs.Function("f_pos", [t], [pos])
    f_vel = cs.Function("f_vel", [t], [vel])
    f_acc = cs.Function("f_acc", [t], [acc])
    f_jerk = cs.Function("f_jerk", [t], [jerk])
    f_snap = cs.Function("f_snap", [t], [snap])

    # evaluation seems to only work for scalar inputs --> iterate over time vector
    pos_list = []
    vel_list = []
    alin_list = []
    t_adj_list = []
    for t_curr in t_vec:
        t_adj_list.append(f_t_adj(t_curr).full().squeeze())
        pos_list.append(f_pos(t_curr).full().squeeze())
        vel_list.append(f_vel(t_curr).full().squeeze())
        alin_list.append(f_acc(t_curr).full().squeeze())

    t_adj_np = np.array(t_adj_list)
    pos_np = np.array(pos_list)
    vel_np = np.array(vel_list)
    alin_np = np.array(alin_list)

    if debug:
        plt.plot(t_adj_np)
        plt.show()

    trajectory, motor_inputs, t_vec = compute_full_traj(
        quad, t_vec, pos_np, vel_np, alin_np
    )

    return trajectory, motor_inputs, t_vec


def load_prepare_trajectory(base_dir, dt, speed_factor, test=False):
    """
    speed factor: between 0 and 1, 0.6 would mean that it's going at 0.6 of the
    actual speed (but discrete steps! if dt=0.05 then speed_factor can only be
    0.8, 0.6, 0.4, etc)
    """
    # FOR TESTING: original version
    # quad = Quad(10.0)
    # arena_bound_max = np.array([6.5, 10, 10])  # np.array([8.0, 5.0, 5.0]) #
    # arena_bound_min = np.array([-6.5, -10, 0])
    # trajectory, _, _ = compute_random_trajectory(
    #     quad, arena_bound_max, arena_bound_min, .9, .7, .7,
    #     10, 0.01, seed=np.random.randint(10000)
    # )
    folder = "test" if test else "train"
    data_list = os.listdir(os.path.join(base_dir, folder))
    rand_traj = np.random.choice(data_list)
    trajectory = np.load(os.path.join(base_dir, folder, rand_traj))

    # dt for trajectory generation is 0.01, then transform back
    take_every_nth = int(dt / 0.01 * speed_factor)
    assert np.isclose(take_every_nth, dt / 0.01 * speed_factor)
    taken_every = trajectory[::take_every_nth, :]

    # transform to euler angels
    quaternions = taken_every[:, 3:7]
    euler_angles = np.array([quaternion_to_euler(q) for q in quaternions])
    # # add in stacking below: euler_angles, taken_every[:, 16:19] (av)
    # stack position, euler angles, velocity, body rates

    # only use pos and vel
    transformed_ref = np.hstack(
        (
            taken_every[:, :3],
            euler_angles * speed_factor,
            taken_every[:, 7:10] * speed_factor * 2,
        )
    )
    # print("transformed shape", transformed_ref.shape)
    return transformed_ref


def make_dataset(num_traj):
    config = {
        "duration": 10,
        "train_split": 0.9,
        "freq_x": 0.9,
        "freq_y": 0.7,
        "freq_z": 0.7,
        "out_dir": "/home/wangzimo/VTT/VTT/aerial_gym/data",
    }

    cutoff = int(num_traj * config["train_split"])
    rand_nums = np.random.permutation(num_traj)
    train_rand_states = rand_nums[:cutoff]
    test_rand_states = rand_nums[cutoff:]

    quad = Quad(10.0)

    # the arena bounds
    arena_bound_max = np.array([8, 8, 10])
    arena_bound_min = np.array([-8, -8, 0])

    for out_dir in ["train", "test"]:
        out_path = os.path.join(config["out_dir"], out_dir)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

    for rand_states, train_test_dir in zip(
        [train_rand_states, test_rand_states], ["train", "test"]
    ):
        out_path = os.path.join(config["out_dir"], train_test_dir)
        for rand in tqdm(rand_states):
            # compute trajectory
            trajectory, _, _ = compute_random_trajectory(
                quad,
                arena_bound_max,
                arena_bound_min,
                config["freq_x"],
                config["freq_y"],
                config["freq_z"],
                config["duration"],
                0.005,
                seed=rand+114,
            )
            np.save(os.path.join(out_path, f"traj_{rand}.npy"), trajectory[:, :10])

    traj_len = len(trajectory)
    config["traj_len"] = traj_len

    with open(os.path.join(config["out_dir"], "config.json"), "w") as outfile:
        json.dump(config, outfile)

def full_state_training_data(
    len_data, ref_length=5, dt=0.01, speed_factor=1, **kwargs
):
    """
    Use trajectory generation of Elia to generate random trajectories and then
    position the drone randomly around the start
    Arguments:
        reset_strength: how much the drone diverges from its desired state
    """
    ref_size = 9
    sample_freq = ref_length * 2
    # TODO: might want to sample less frequently
    drone_states = np.zeros((len_data + 200, 12))
    ref_states = np.zeros((len_data + 200, ref_length, ref_size))

    counter = 0
    while counter < len_data:
        traj = load_prepare_trajectory(
            "/home/wangzimo/VTT/VTT/aerial_gym/data", dt, speed_factor, test=0
        )[:, :ref_size]
        traj_cut = traj[:-(ref_length + 1)]
        # select every xth sample as the current drone state
        selected_starts = traj_cut[::sample_freq, :]
        nr_states_added = len(selected_starts)

        full_drone_state = np.hstack(
            (selected_starts, np.zeros((len(selected_starts), 3)))
        )
        # add drone states
        drone_states[counter:counter + nr_states_added, :] = full_drone_state
        # add ref states
        for i in range(1, ref_length + 1):
            ref_states[counter:counter + nr_states_added,
                       i - 1] = (traj[i::sample_freq])[:nr_states_added]

        counter += nr_states_added

    return drone_states[:len_data], ref_states[:len_data]

def load_training_data(base_dir, dt=0.01, speed_factor=1, test=False):
    """
    speed factor: between 0 and 1, 0.6 would mean that it's going at 0.6 of the
    actual speed (but discrete steps! if dt=0.05 then speed_factor can only be
    0.8, 0.6, 0.4, etc)
    """
    # FOR TESTING: original version
    # quad = Quad(10.0)
    # arena_bound_max = np.array([6.5, 10, 10])  # np.array([8.0, 5.0, 5.0]) #
    # arena_bound_min = np.array([-6.5, -10, 0])
    # trajectory, _, _ = compute_random_trajectory(
    #     quad, arena_bound_max, arena_bound_min, .9, .7, .7,
    #     10, 0.01, seed=np.random.randint(10000)
    # )
    folder = "test" if test else "train"
    data_list = os.listdir(os.path.join(base_dir, folder))
    traj_list = []
    for data in data_list:

        trajectory = np.load(os.path.join(base_dir, folder, data))

        # dt for trajectory generation is 0.01, then transform back
        take_every_nth = int(dt / 0.01 * speed_factor)
        assert np.isclose(take_every_nth, dt / 0.01 * speed_factor)
        taken_every = trajectory[::take_every_nth, :]

        # transform to euler angels
        quaternions = taken_every[:, 3:7]
        euler_angles = np.array([quaternion_to_euler(q) for q in quaternions])
        # # add in stacking below: euler_angles, taken_every[:, 16:19] (av)
        # stack position, euler angles, velocity, body rates

        # only use pos and vel
        transformed_ref = np.hstack(
            (
                taken_every[:, :3],
                euler_angles * speed_factor,
                taken_every[:, 7:10] * speed_factor * 2,
            )
        )
        traj_list.append(transformed_ref)
    # print("transformed shape", transformed_ref.shape)
    return traj_list