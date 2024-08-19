import torch.nn as nn
import torch
from torch import tensor
import torch.nn.functional as F

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
    
    return 0.5 * loss_direction + 0.3 * loss_h + 0.1 * loss_speed + 0.1 * loss_ori, loss_direction, loss_speed, loss_ori, loss_h