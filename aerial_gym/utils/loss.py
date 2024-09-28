import torch.nn as nn
import torch
import torch.nn.functional as F
import copy

class Loss:
    def __init__(self, batch_size, device):
        self.device = device
        self.batch_size = batch_size

        self.direction = torch.zeros(batch_size, device=device)
        self.speed = torch.zeros(batch_size, device=device)
        self.h = torch.zeros(batch_size, device=device)
        self.acc = torch.zeros(batch_size, device=device)
        self.jerk = torch.zeros(batch_size, device=device)
        self.ori = torch.zeros(batch_size, device=device)

    def reset(self, reset_idx):
        self.direction[reset_idx] = 0
        self.speed[reset_idx] = 0
        self.h[reset_idx] = 0
        self.acc[reset_idx] = 0
        self.jerk[reset_idx] = 0
        self.ori[reset_idx] = 0
    

def space_lossVer5(loss:Loss, quad_state, acceleration, last_acceleration, tar_state, tar_h, tar_ori, step, dt):
    """
    Refering to Back to Newton...

    """
    vel = quad_state[:, 6:9]
    ori = quad_state[:, 3:6]

    
    z_coords = torch.full((quad_state.size(0), 1), tar_h, dtype=quad_state.dtype, device=quad_state.device)
    tar_pos = torch.cat((tar_state[:, :2].clone(), z_coords), dim=1)
    
    hor_dis = (tar_pos[:, :2].clone() - quad_state[:, :2].clone())
    
    norm_hor_dis = torch.norm(hor_dis, dim=1, p=2)
    norm_hor_vel = torch.norm(vel[:, :2].clone(), dim=1, p=2)
    
    tmp_norm_dis = torch.clamp(norm_hor_dis, max=5)

    new_loss = Loss(loss.batch_size, loss.device)

    loss_direction = (1 - F.cosine_similarity(hor_dis, vel[:, :2].clone())) * tmp_norm_dis
    new_loss.direction = (loss.direction * step + loss_direction) / (step + 1)

    loss_speed = torch.abs(tmp_norm_dis - norm_hor_vel)
    new_loss.speed = (loss.speed * step + loss_speed) / (step + 1)
    
    loss_h = torch.abs(quad_state[:, 2] - tar_h)
    new_loss.h = (loss.h * step + loss_h) / (step + 1)
    
    loss_ori = torch.norm(tar_ori - ori, dim=1, p=2)
    new_loss.ori = (loss.ori * step + loss_ori) / (step + 1)

    loss_acc = torch.norm(acceleration.clone(), dim=1, p=2)
    new_loss.acc = (loss.acc * step + loss_acc) / (step + 1)

    loss_jerk = torch.norm((acceleration.clone() - last_acceleration.clone()) / dt, dim=1, p=2)
    new_loss.jerk = (loss.jerk * step + loss_jerk) / (step + 1)

    # print(f"loss.direction[-1]:", loss.direction[-1])
    loss_final = 0.4 * new_loss.direction + 0.299 * new_loss.speed + 0.2 * new_loss.h + 0.1 * new_loss.ori + 0.0009 * new_loss.acc + 0.0001 * new_loss.jerk
    # loss_final = loss_direction
    # print(f"loss_final shape:{loss_final.shape}, {0.4 * loss.direction[-1]}")
    # loss_final = new_loss.speed

    return loss_final, new_loss



def space_lossVer4(quad_state, tar_state, tar_pos, tar_h, tar_ori):
    """
    In last version model doesn't converge as well. Tring to delet latent intend part
    """
    pos = quad_state[:, :3]
    ori = quad_state[:, 3:6]
    vel = quad_state[:, 6:9]

    tar_pos = tar_state[:, :3]
    tar_vel = tar_state[:, 6:9]

    z_coords = torch.full((quad_state.size(0), 1), tar_h, dtype=quad_state.dtype, device=quad_state.device)
    tar_pos = torch.cat((tar_pos[:, :2], z_coords), dim=1)
    
    hor_dis = (tar_pos[:, :2] - quad_state[:, :2])
    norm_hor_dis = torch.norm(hor_dis, dim=1, p=2)
    # norm_hor_vel = torch.norm(vel[:, :2], dim=1, p=2)
    tmp_norm_dis = torch.clamp(norm_hor_dis, max=5)

    # loss_distance = torch.norm(pos[:, :2] - tar_pos[:, :2], dim=1, p=2)
    loss_direction = (1 - F.cosine_similarity(hor_dis, vel[:, :2])) * tmp_norm_dis
    loss_speed = torch.norm(vel[:, :2] - tar_vel[:, :2], dim=1, p=2)
    loss_h = torch.abs(quad_state[:, 2] - tar_h)
    loss_ori = torch.norm(tar_ori - ori, dim=1, p=2)
    # print("Predicted relative distance size:", predicted_rel_dis.size())
    # print("Real relative distance size:", real_rel_dis.size())
    # print("loss_intent size:", loss_intent.size())

    return 0.5 * loss_direction + 0.05 * loss_speed + 0.3 * loss_h + 0.15 * loss_ori, loss_direction, loss_speed, loss_h, loss_ori


def space_lossVer3(quad_state, tar_state, predicted_rel_dis, real_rel_dis, tar_pos, tar_h, tar_ori, criterion):
    """
    In last version model doesn't converge. Replace distance loss with velocity direction loss as experiment.
    """
    pos = quad_state[:, :3]
    ori = quad_state[:, 3:6]
    vel = quad_state[:, 6:9]

    tar_pos = tar_state[:, :3]
    tar_vel = tar_state[:, 6:9]

    z_coords = torch.full((quad_state.size(0), 1), tar_h, dtype=quad_state.dtype, device=quad_state.device)
    tar_pos = torch.cat((tar_pos[:, :2], z_coords), dim=1)
    
    hor_dis = (tar_pos[:, :2] - quad_state[:, :2])
    norm_hor_dis = torch.norm(hor_dis, dim=1, p=2)
    # norm_hor_vel = torch.norm(vel[:, :2], dim=1, p=2)
    tmp_norm_dis = torch.clamp(norm_hor_dis, max=5)

    # loss_distance = torch.norm(pos[:, :2] - tar_pos[:, :2], dim=1, p=2)
    loss_direction = (1 - F.cosine_similarity(hor_dis, vel[:, :2])) * tmp_norm_dis
    loss_speed = torch.norm(vel[:, :2] - tar_vel[:, :2], dim=1, p=2)
    loss_h = torch.abs(quad_state[:, 2] - tar_h)
    loss_ori = torch.norm(tar_ori - ori, dim=1, p=2)
    # print("Predicted relative distance size:", predicted_rel_dis.size())
    # print("Real relative distance size:", real_rel_dis.size())
    loss_intent = torch.mean(criterion(predicted_rel_dis, real_rel_dis), dim=1)
    loss_intent = torch.mean(loss_intent, dim=1)
    # print("loss_intent size:", loss_intent.size())

    return 0.4 * loss_direction + 0.05 * loss_speed + 0.3 * loss_h + 0.15 * loss_ori + 0.1 * loss_intent, loss_direction, loss_speed, loss_h, loss_ori, loss_intent


def space_lossVer2(quad_state, tar_state, predicted_rel_dis, real_rel_dis, tar_pos, tar_h, tar_ori, criterion):
    """
    Added speed constrain to prevent drone from flying ahead of the target
    """
    pos = quad_state[:, :3]
    ori = quad_state[:, 3:6]
    vel = quad_state[:, 6:9]

    tar_pos = tar_state[:, :3]
    tar_vel = tar_state[:, 6:9]

    z_coords = torch.full((quad_state.size(0), 1), tar_h, dtype=quad_state.dtype, device=quad_state.device)
    tar_pos = torch.cat((tar_pos[:, :2], z_coords), dim=1)
    
    hor_dis = (tar_pos[:, :2] - quad_state[:, :2])
    norm_hor_dis = torch.norm(hor_dis, dim=1, p=2)
    # norm_hor_vel = torch.norm(vel[:, :2], dim=1, p=2)
    tmp_norm_dis = torch.clamp(norm_hor_dis, max=5)

    loss_distance = torch.norm(pos[:, :2] - tar_pos[:, :2], dim=1, p=2)
    loss_speed = torch.norm(vel[:, :2] - tar_vel[:, :2], dim=1, p=2)
    loss_h = torch.abs(quad_state[:, 2] - tar_h)
    loss_ori = torch.norm(tar_ori - ori, dim=1, p=2)
    # print("Predicted relative distance size:", predicted_rel_dis.size())
    # print("Real relative distance size:", real_rel_dis.size())
    loss_intent = torch.mean(criterion(predicted_rel_dis, real_rel_dis), dim=1)
    loss_intent = torch.mean(loss_intent, dim=1)
    # print("loss_intent size:", loss_intent.size())

    return 0.2 * loss_distance + 0.25 * loss_speed + 0.3 * loss_h + 0.15 * loss_ori + 0.1 * loss_intent, loss_distance, loss_speed, loss_h, loss_ori, loss_intent


def space_lossVer1(quad_state, tar_state, predicted_rel_dis, real_rel_dis, tar_pos, tar_h, tar_ori, criterion):
    """
    Added speed constrain to prevent drone from flying ahead of the target
    """
    pos = quad_state[:, :3]
    ori = quad_state[:, 3:6]
    vel = quad_state[:, 6:9]

    tar_pos = tar_state[:, :3]
    tar_vel = tar_state[:, 6:9]

    z_coords = torch.full((quad_state.size(0), 1), tar_h, dtype=quad_state.dtype, device=quad_state.device)
    tar_pos = torch.cat((tar_pos[:, :2], z_coords), dim=1)
    
    hor_dis = (tar_pos[:, :2] - quad_state[:, :2])
    norm_hor_dis = torch.norm(hor_dis, dim=1, p=2)
    # norm_hor_vel = torch.norm(vel[:, :2], dim=1, p=2)
    tmp_norm_dis = torch.clamp(norm_hor_dis, max=5)

    loss_distance = torch.norm(pos[:, :2] - tar_pos[:, :2])
    loss_direction = (1 - F.cosine_similarity(hor_dis, vel[:, :2])) * tmp_norm_dis
    loss_speed = torch.norm(vel[:, :2] - tar_vel[:, :2], dim=1, p=2)
    loss_h = torch.abs(quad_state[:, 2] - tar_h)
    loss_ori = torch.norm(tar_ori - ori, dim=1, p=2)
    # print("Predicted relative distance size:", predicted_rel_dis.size())
    # print("Real relative distance size:", real_rel_dis.size())
    loss_intent = torch.mean(criterion(predicted_rel_dis, real_rel_dis), dim=1)
    loss_intent = torch.mean(loss_intent, dim=1)
    # print("loss_intent size:", loss_intent.size())

    return 0.15 * loss_distance + 0.25 * loss_direction + 0.15 * loss_speed + 0.3 * loss_h + 0.05 * loss_ori + 0.1 * loss_intent, loss_distance, loss_direction, loss_speed, loss_h, loss_ori, loss_intent


def space_lossVer0(quad_state, predicted_rel_dis, real_rel_dis, tar_pos, tar_h, tar_ori, criterion):
    """
    Drone easily fly ahead
    """
    
    
    ori = quad_state[:, 3:6]
    vel = quad_state[:, 6:9]

    z_coords = torch.full((quad_state.size(0), 1), tar_h, dtype=quad_state.dtype, device=quad_state.device)
    tar_pos = torch.cat((tar_pos[:, :2], z_coords), dim=1)
    
    hor_dis = (tar_pos[:, :2] - quad_state[:, :2])
    norm_hor_dis = torch.norm(hor_dis, dim=1, p=2)
    tmp_norm_dis = torch.clamp(norm_hor_dis, max=5)

    loss_direction = (1 - F.cosine_similarity(hor_dis, vel[:, :2])) * tmp_norm_dis
    loss_h = torch.abs(quad_state[:, 2] - tar_h)
    loss_ori = torch.norm(tar_ori - ori, dim=1, p=2)
    # print("Predicted relative distance size:", predicted_rel_dis.size())
    # print("Real relative distance size:", real_rel_dis.size())
    # print(criterion(predicted_rel_dis, real_rel_dis).size())
    loss_intent = torch.mean(criterion(predicted_rel_dis, real_rel_dis), dim=1)
    loss_intent = torch.mean(loss_intent, dim=1)
    # print("loss_intent size:", loss_intent.size())

    return 0.45 * loss_direction + 0.4 * loss_h + 0.05 * loss_ori + 0.1 * loss_intent, loss_direction, loss_h, loss_ori, loss_intent


def acc_loss(quad_state, tar_pos, criterion):
    acc = quad_state[:, 6:8]
    # print(tar_pos.shape, quad_state.shape)
    dis = (tar_pos[:, :2] - quad_state[:, :2])
    return criterion(acc, dis)

def acch_loss(quad_state, tar_pos, tar_height, criterion):
    acc = quad_state[:, 6:9]
    dis_hoz = (tar_pos[:, :2] - quad_state[:, :2])
    dis_ver = torch.tensor(tar_height) - quad_state[:, 2]
    dis_ver = torch.unsqueeze(dis_ver, dim=1)
    # print(dis_hoz.shape, dis_ver.shape)
    dis = torch.cat((dis_hoz, dis_ver), dim=1)
    return criterion(acc, dis)


def pav_loss(quad_state, tar_pos, tar_h, criterion):
    vel = quad_state[:, 6:9]
    acc = quad_state[:, 9:12]

    z_coords = torch.full((quad_state.size(0), 1), tar_h, dtype=quad_state.dtype, device=quad_state.device)
    
    tar_pos = tar_pos[:, :2]
    tar_pos = torch.cat((tar_pos, z_coords), dim=1)
    # print(tar_pos.shape, quad_state[:, :3].shape)
    dis = (tar_pos - quad_state[:, :3])
    
    loss_vel = criterion(vel, dis)
    loss_acc = criterion(acc, dis)
    loss_dis = criterion(tar_pos, quad_state[:, :3])
    
    loss = 0.8 * loss_dis + 0.15 * loss_vel + 0.05 * loss_acc
    return loss

def velh_loss(quad_state, tar_pos, tar_h):
    vel = quad_state[:, 6:9]
    
    z_coords = torch.full((quad_state.size(0), 1), tar_h, dtype=quad_state.dtype, device=quad_state.device)
    tar_pos = torch.cat((tar_pos[:, :2], z_coords), dim=1)
    dis = (tar_pos - quad_state[:, :3])
    
    loss1 = torch.norm(vel[:, :2] - dis[:, :2], dim=1, p=2)
    loss2 = torch.abs(vel[:, 2] - dis[:, 2]) 
    loss = loss1 + 0.01 * loss2
    return loss

def velh_lossVer2(quad_state, tar_pos, tar_h, criterion):
    vel = quad_state[:, 6:9]
    
    z_coords = torch.full((quad_state.size(0), 1), tar_h, dtype=quad_state.dtype, device=quad_state.device)
    tar_pos = torch.cat((tar_pos[:, :2], z_coords), dim=1)
    dis = (tar_pos - quad_state[:, :3])
    # dis[:, 2] *= 0.1
    
    return criterion(dis, vel)

def velh_lossVer3(quad_state, tar_pos, tar_h):
    vel = quad_state[:, 6:9]
    
    z_coords = torch.full((quad_state.size(0), 1), tar_h, dtype=quad_state.dtype, device=quad_state.device)
    tar_pos = torch.cat((tar_pos[:, :2], z_coords), dim=1)
    
    dis = (tar_pos - quad_state[:, :3])
    # dis[:, 2] *= 0.4
    
    norm_dis = torch.norm(dis, dim=1, p=2)
    norm_vel = torch.norm(vel, dim=1, p=2)
    
    loss_direction = (1 - F.cosine_similarity(dis, vel)) * 5
    tmp_norm_dis = torch.clamp(norm_dis, max=10)
    loss_speed = torch.abs(tmp_norm_dis - norm_vel)
    
    return loss_direction + loss_speed, loss_direction, loss_speed

def velh_lossVer4(quad_state, tar_pos, tar_h, tar_ori):
    vel = quad_state[:, 6:9]
    ori = quad_state[:, 3:6]
    
    z_coords = torch.full((quad_state.size(0), 1), tar_h, dtype=quad_state.dtype, device=quad_state.device)
    tar_pos = torch.cat((tar_pos[:, :2], z_coords), dim=1)
    
    dis = (tar_pos - quad_state[:, :3])
    dis[:, 2] *= 4
    
    norm_dis = torch.norm(dis, dim=1, p=2)
    norm_vel = torch.norm(vel, dim=1, p=2)
    
    tmp_norm_dis = torch.clamp(norm_dis, max=5)
    
    loss_direction = (1 - F.cosine_similarity(dis, vel)) * tmp_norm_dis
    
    loss_speed = torch.abs(tmp_norm_dis - norm_vel)
    
    loss_ori = torch.norm(tar_ori - ori, dim=1, p=2)
    
    return 0.8 * loss_direction + 0.1 * loss_speed + 0.1 * loss_ori, loss_direction, loss_speed, loss_ori


def velh_lossVer5(quad_state, tar_pos, tar_h, tar_ori):
    vel = quad_state[:, 6:9]
    ori = quad_state[:, 3:6]
    
    z_coords = torch.full((quad_state.size(0), 1), tar_h, dtype=quad_state.dtype, device=quad_state.device)
    tar_pos = torch.cat((tar_pos[:, :2], z_coords), dim=1)
    
    hor_dis = (tar_pos[:, :2] - quad_state[:, :2])
    
    norm_hor_dis = torch.norm(hor_dis, dim=1, p=2)
    norm_hor_vel = torch.norm(vel[:, :2], dim=1, p=2)
    
    tmp_norm_dis = torch.clamp(norm_hor_dis, max=5)
    
    loss_direction = (1 - F.cosine_similarity(hor_dis, vel[:, :2])) * tmp_norm_dis
    
    loss_speed = torch.abs(tmp_norm_dis - norm_hor_vel)
    
    loss_h = torch.abs(quad_state[:, 2] - tar_h)
    
    loss_ori = torch.norm(tar_ori - ori, dim=1, p=2)
    
    return 0.5 * loss_direction + 1.3 * loss_h + 0.1 * loss_speed + 0.1 * loss_ori, loss_direction, loss_speed, loss_ori, loss_h


def velh_lossVer6(quad_state, tar_pos, tar_h, tar_ori, action):
    """
    Change height loss from distance to velocity direction
    """
    vel = quad_state[:, 6:9]
    ori = quad_state[:, 3:6]
    
    z_coords = torch.full((quad_state.size(0), 1), tar_h, dtype=quad_state.dtype, device=quad_state.device)
    tar_pos = torch.cat((tar_pos[:, :2], z_coords), dim=1)
    
    dis = (tar_pos - quad_state[:, :3])
    
    norm_dis = torch.norm(dis, dim=1, p=2)
    norm_hor_dis = torch.norm(dis[:, :2], dim=1, p=2)
    norm_hor_vel = torch.norm(vel[:, :2], dim=1, p=2)
    norm_action = torch.norm(action, dim=1, p=2)
    
    tmp_norm_dis = torch.clamp(norm_dis, max=5)
    tmp_norm_hor_dis = torch.clamp(norm_hor_dis, max=5)
    
    loss_direction = (1 - F.cosine_similarity(dis, vel[:, :3])) * tmp_norm_dis
    
    loss_speed = torch.abs(tmp_norm_hor_dis - norm_hor_vel)
    
    loss_h = torch.abs(quad_state[:, 2] - tar_h)
    
    loss_ori = torch.norm(tar_ori - ori, dim=1, p=2) * norm_action
    # if torch.isnan(loss_ori).any():
    #     print("????????????", loss_ori, ori, tar_ori)


    return 0.97 * loss_direction + 0.02 * loss_speed + 0.01 * loss_ori, loss_direction, loss_speed, loss_ori, loss_h