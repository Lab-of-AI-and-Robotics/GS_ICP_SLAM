import torch
import numpy as np
import cv2
import torch.nn as nn
import copy
import math

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R.t()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = Rt.inverse()
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = C2W.inverse()
    return Rt

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class SharedPoints(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.points = torch.zeros((num_points, 3)).float()
        self.colors = torch.zeros((num_points, 3)).float()
        self.z_values = torch.zeros((num_points)).float()
        self.filter = torch.zeros((num_points)).int()
        self.using_idx = torch.zeros((1)).int()
        self.filter_size = torch.zeros((1)).int()
    
    def input_values(self, new_points, new_colors, new_z_values, new_filter):
        self.using_idx[0] = new_points.shape[0]
        self.points[:self.using_idx[0],:] = new_points
        self.colors[:self.using_idx[0],:] = new_colors
        self.z_values[:self.using_idx[0]] = new_z_values
        
        self.filter_size[0] = new_filter.shape[0]
        self.filter[:self.filter_size[0]] = new_filter

    def get_values(self):
        return  copy.deepcopy(self.points[:self.using_idx[0],:].numpy()),\
                copy.deepcopy(self.colors[:self.using_idx[0],:].numpy()),\
                copy.deepcopy(self.z_values[:self.using_idx[0]].numpy()),\
                copy.deepcopy(self.filter[:self.filter_size[0]].numpy())

class SharedGaussians(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.xyz = torch.zeros((num_points, 3)).float().cuda()
        self.colors = torch.zeros((num_points, 3)).float().cuda()
        self.rots = torch.zeros((num_points, 4)).float().cuda()
        self.scales = torch.zeros((num_points, 3)).float().cuda()
        self.z_values = torch.zeros((num_points)).float().cuda()
        self.trackable_filter = torch.zeros((num_points)).long().cuda()
        self.using_idx = torch.zeros((1)).int().cuda()
        self.filter_size = torch.zeros((1)).int().cuda()

    def input_values(self, new_xyz, new_colors, new_rots, new_scales, new_z_values, new_trackable_filter):
        # on CPU memory
        self.using_idx[0] = new_xyz.shape[0]
        self.xyz[:self.using_idx[0],:] = new_xyz
        self.colors[:self.using_idx[0],:] = new_colors
        self.rots[:self.using_idx[0],:] = new_rots
        self.scales[:self.using_idx[0],:] = new_scales
        self.z_values[:self.using_idx[0]] = new_z_values
        
        self.filter_size[0] = new_trackable_filter.shape[0]
        self.trackable_filter[:self.filter_size[0]] = new_trackable_filter
    
    def get_values(self):
        return  copy.deepcopy(self.xyz[:self.using_idx[0],:]),\
                copy.deepcopy(self.colors[:self.using_idx[0],:]),\
                copy.deepcopy(self.rots[:self.using_idx[0],:]),\
                copy.deepcopy(self.scales[:self.using_idx[0],:]),\
                copy.deepcopy(self.z_values[:self.using_idx[0]]),\
                copy.deepcopy(self.trackable_filter[:self.filter_size[0]])

class SharedTargetPoints(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.num_points = num_points
        self.xyz = torch.zeros((num_points, 3)).float()
        self.rots = torch.zeros((num_points, 4)).float()
        self.scales = torch.zeros((num_points, 3)).float()
        self.using_idx = torch.zeros((1)).int()

    def input_values(self, new_xyz, new_rots, new_scales):
        self.using_idx[0] = new_xyz.shape[0]
        if self.using_idx[0]>self.num_points:
            print("Too many target points")
        self.xyz[:self.using_idx[0],:] = new_xyz
        self.rots[:self.using_idx[0],:] = new_rots
        self.scales[:self.using_idx[0],:] = new_scales
    
    def get_values_tensor(self):
        return  copy.deepcopy(self.xyz[:self.using_idx[0],:]),\
                copy.deepcopy(self.rots[:self.using_idx[0],:]),\
                copy.deepcopy(self.scales[:self.using_idx[0],:])

    def get_values_np(self):
        return  copy.deepcopy(self.xyz[:self.using_idx[0],:].numpy()),\
                copy.deepcopy(self.rots[:self.using_idx[0],:].numpy()),\
                copy.deepcopy(self.scales[:self.using_idx[0],:].numpy())

class SharedCam(nn.Module):
    def __init__(self, FoVx, FoVy, image, depth_image,
                 cx, cy, fx, fy,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
        super().__init__()
        self.cam_idx = torch.zeros((1)).int()
        self.R = torch.eye(3,3).float()
        self.t = torch.zeros((3)).float()
        self.FoVx = torch.tensor([FoVx])
        self.FoVy = torch.tensor([FoVy])
        self.image_width = torch.tensor([image.shape[1]])
        self.image_height = torch.tensor([image.shape[0]])
        self.cx = torch.tensor([cx])
        self.cy = torch.tensor([cy])
        self.fx = torch.tensor([fx])
        self.fy = torch.tensor([fy])
        
        self.original_image = torch.from_numpy(image).float().permute(2,0,1)/255
        # rgb_level_1 = cv2.resize(image, (self.image_width//2, self.image_height//2))
        # rgb_level_2 = cv2.resize(image, (self.image_width//4, self.image_height//4))
        # self.rgb_level_1 = torch.from_numpy(rgb_level_1).float().cuda().permute(2,0,1)/255
        # self.rgb_level_2 = torch.from_numpy(rgb_level_2).float().cuda().permute(2,0,1)/255
        
        self.original_depth_image = torch.from_numpy(depth_image).float().unsqueeze(0)
        # depth_level_1 = cv2.resize(depth_image, (self.image_width//2, self.image_height//2), interpolation=cv2.INTER_NEAREST)
        # depth_level_2 = cv2.resize(depth_image, (self.image_width//4, self.image_height//4), interpolation=cv2.INTER_NEAREST)
        # self.depth_level_1 = torch.from_numpy(depth_level_1).float().unsqueeze(0).cuda()
        # self.depth_level_2 = torch.from_numpy(depth_level_2).float().unsqueeze(0).cuda()
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = getWorld2View2(self.R, self.t, trans, scale).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
    def update_matrix(self):
        self.world_view_transform[:,:] = getWorld2View2(self.R, self.t, self.trans, self.scale).transpose(0, 1)
        # self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform[:,:] = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center[:] = self.world_view_transform.inverse()[3, :3]
    
    def setup_cam(self, R, t, rgb_img, depth_img):
        # Set pose, projection matrix
        self.R[:,:] = torch.from_numpy(R)
        self.t[:] = torch.from_numpy(t)
        self.update_matrix()
        # Update image
        self.original_image[:,:,:] = torch.from_numpy(rgb_img).float().permute(2,0,1)/255
        self.original_depth_image[:,:,:] = torch.from_numpy(depth_img).float().unsqueeze(0)
    
    def on_cuda(self):
        self.world_view_transform = self.world_view_transform.cuda()
        self.projection_matrix = self.projection_matrix.cuda()
        self.full_proj_transform = self.full_proj_transform.cuda()
        self.camera_center = self.camera_center.cuda()
        
        self.original_image = self.original_image.cuda()
        self.original_depth_image = self.original_depth_image.cuda()
        


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