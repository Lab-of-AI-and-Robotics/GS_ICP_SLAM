#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import cv2
import torch.nn as nn
import time

class Camera():
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, depth_image, gt_alpha_mask,
                 image_name, depth_image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.depth_image_name = depth_image_name
        
        self.original_image = image.clamp(0.0, 1.0).to('cuda:1')
        self.original_depth_image = depth_image.to('cuda:1')
        
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    def update(self):
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MappingCams(nn.Module):
    def __init__(self):
        super().__init__()
        self.cams = []
        self.test = torch.tensor([1,2,3])
    def add_new_camera(self, newcam):
        self.cams.append(newcam)
        self.cams[-1].share_memory()

class MappingCam(nn.Module):
    def __init__(self, cam_idx, R, t, FoVx, FoVy, image, depth_image,
                 cx, cy, fx, fy,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super().__init__()
        self.cam_idx = cam_idx
        self.R = R
        self.t = t
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = image.shape[1]
        self.image_height = image.shape[0]
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.last_loss = 0.
        
        self.original_image = torch.from_numpy(image).float().cuda().permute(2,0,1)/255
        # rgb_level_1 = cv2.resize(image, (self.image_width//2, self.image_height//2))
        # rgb_level_2 = cv2.resize(image, (self.image_width//4, self.image_height//4))
        # self.rgb_level_1 = torch.from_numpy(rgb_level_1).float().cuda().permute(2,0,1)/255
        # self.rgb_level_2 = torch.from_numpy(rgb_level_2).float().cuda().permute(2,0,1)/255
        
        self.original_depth_image = torch.from_numpy(depth_image).float().unsqueeze(0).cuda()
        # depth_level_1 = cv2.resize(depth_image, (self.image_width//2, self.image_height//2), interpolation=cv2.INTER_NEAREST)
        # depth_level_2 = cv2.resize(depth_image, (self.image_width//4, self.image_height//4), interpolation=cv2.INTER_NEAREST)
        # self.depth_level_1 = torch.from_numpy(depth_level_1).float().unsqueeze(0).cuda()
        # self.depth_level_2 = torch.from_numpy(depth_level_2).float().unsqueeze(0).cuda()
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, t, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    def update(self):
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.t, self.trans, self.scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

