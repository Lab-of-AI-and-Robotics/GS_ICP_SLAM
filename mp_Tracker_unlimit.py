import os
import torch
import torch.multiprocessing as mp
import torch.multiprocessing
from random import randint
import sys
import cv2
import numpy as np
import open3d as o3d
import pygicp
import time
from scipy.spatial.transform import Rotation
import rerun as rr
sys.path.append(os.path.dirname(__file__))
from arguments import SLAMParameters
from utils.traj_utils import TrajManager
from gaussian_renderer import render, render_2, network_gui
from tqdm import tqdm


class Tracker(SLAMParameters):
    def __init__(self, slam):
        super().__init__()
        self.dataset_path = slam.dataset_path
        self.output_path = slam.output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.verbose = slam.verbose
        self.keyframe_th = slam.keyframe_th
        self.knn_max_distance = slam.knn_max_distance
        self.overlapped_th = slam.overlapped_th
        self.overlapped_th2 = slam.overlapped_th2
        self.downsample_rate = slam.downsample_rate
        self.test = slam.test
        
        self.camera_parameters = slam.camera_parameters
        self.W = slam.W
        self.H = slam.H
        self.fx = slam.fx
        self.fy = slam.fy
        self.cx = slam.cx
        self.cy = slam.cy
        self.depth_scale = slam.depth_scale
        self.depth_trunc = slam.depth_trunc
        self.cam_intrinsic = np.array([[self.fx, 0., self.cx],
                                       [0., self.fy, self.cy],
                                       [0.,0.,1]])
        
        self.viewer_fps = slam.viewer_fps
        self.keyframe_freq = slam.keyframe_freq
        self.max_correspondence_distance = slam.max_correspondence_distance
        self.reg = pygicp.FastGICP()
        
        # Camera poses
        self.trajmanager = TrajManager(self.camera_parameters[8], self.dataset_path)
        self.poses = [self.trajmanager.gt_poses[0]]
        # Keyframes(added to map gaussians)
        self.last_t = time.time()
        self.iteration_images = 0
        self.end_trigger = False
        self.covisible_keyframes = []
        self.new_target_trigger = False
        
        self.cam_t = []
        self.cam_R = []
        self.points_cat = []
        self.colors_cat = []
        self.rots_cat = []
        self.scales_cat = []
        self.trackable_mask = []
        self.from_last_tracking_keyframe = 0
        self.from_last_mapping_keyframe = 0
        self.scene_extent = 2.5
        
        self.downsample_idxs, self.x_pre, self.y_pre = self.set_downsample_filter(self.downsample_rate)

        # Share
        self.train_iter = 0
        self.mapping_losses = []
        self.new_keyframes = []
        self.gaussian_keyframe_idxs = []

        self.shared_cam = slam.shared_cam
        self.shared_new_points = slam.shared_new_points
        self.shared_new_gaussians = slam.shared_new_gaussians
        self.shared_target_gaussians = slam.shared_target_gaussians
        self.end_of_dataset = slam.end_of_dataset
        self.is_tracking_keyframe_shared = slam.is_tracking_keyframe_shared
        self.is_mapping_keyframe_shared = slam.is_mapping_keyframe_shared
        self.target_gaussians_ready = slam.target_gaussians_ready
        self.new_points_ready = slam.new_points_ready
        self.final_pose = slam.final_pose
        self.demo = slam.demo
        self.is_mapping_process_started = slam.is_mapping_process_started
    
    def run(self):
        self.tracking()
    
    def tracking(self):
        tt = torch.zeros((1,1)).float().cuda()
        
        self.rgb_images, self.depth_images = self.get_images(f"{self.dataset_path}/images")
        self.num_images = len(self.rgb_images)
        self.reg.set_max_correspondence_distance(self.max_correspondence_distance)
        self.reg.set_max_knn_distance(self.knn_max_distance)
        if_mapping_keyframe = False

        # print("Waiting for mapping process to be prepared")
        # while not self.is_mapping_process_started[0]:
        #     time.sleep(0.01)

        self.total_start_time = time.time()
        pbar = tqdm(total=self.num_images)

        for ii in range(self.num_images):
            current_image = self.rgb_images.pop(0)
            depth_image = self.depth_images.pop(0)
            current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)
            
            # Make pointcloud
            points, colors, z_values, trackable_filter = self.downsample_and_make_pointcloud2(depth_image, current_image)
            # GICP
            if self.iteration_images == 0:
                current_pose = self.poses[-1]
                
                # Update Camera pose #
                current_pose = np.linalg.inv(current_pose)
                T = current_pose[:3,3]
                R = current_pose[:3,:3].transpose()
                
                # transform current points
                points = np.matmul(R, points.transpose()).transpose() - np.matmul(R, T)
                # Set initial pointcloud to target points
                self.reg.set_input_target(points)
                
                num_trackable_points = trackable_filter.shape[0]
                input_filter = np.zeros(points.shape[0], dtype=np.int32)
                input_filter[(trackable_filter)] = [range(1, num_trackable_points+1)]
                
                self.reg.set_target_filter(num_trackable_points, input_filter)
                self.reg.calculate_target_covariance_with_filter()

                rots = self.reg.get_target_rotationsq()
                scales = self.reg.get_target_scales()
                rots = np.reshape(rots, (-1,4))
                scales = np.reshape(scales, (-1,3))
                
                # Assign first gaussian to shared memory
                self.shared_new_gaussians.input_values(torch.tensor(points), torch.tensor(colors), 
                                                       torch.tensor(rots), torch.tensor(scales), 
                                                       torch.tensor(z_values), torch.tensor(trackable_filter))
                
                # Add first keyframe
                depth_image = depth_image.astype(np.float32)/self.depth_scale
                self.shared_cam.setup_cam(R, T, current_image, depth_image)
                self.shared_cam.cam_idx[0] = self.iteration_images
                
                self.is_tracking_keyframe_shared[0] = 1
                
                while self.demo[0]:
                    time.sleep(1e-15)
                    self.total_start_time = time.time()
            else:
                self.reg.set_input_source(points)
                num_trackable_points = trackable_filter.shape[0]
                input_filter = np.zeros(points.shape[0], dtype=np.int32)
                input_filter[(trackable_filter)] = [range(1, num_trackable_points+1)]
                self.reg.set_source_filter(num_trackable_points, input_filter)
                
                initial_pose = self.poses[-1]

                current_pose = self.reg.align(initial_pose)
                self.poses.append(current_pose)

                # Update Camera pose #
                current_pose = np.linalg.inv(current_pose)
                T = current_pose[:3,3]
                R = current_pose[:3,:3].transpose()

                # transform current points
                points = np.matmul(R, points.transpose()).transpose() - np.matmul(R, T)
                # Use only trackable points when tracking
                target_corres, distances = self.reg.get_source_correspondence() # get associated points source points
                
                # Keyframe selection #
                # Tracking keyframe
                len_corres = len(np.where(distances<self.overlapped_th)[0]) # 5e-4 self.overlapped_th
                
                if  (self.iteration_images >= self.num_images-1 \
                    or len_corres/distances.shape[0] < self.keyframe_th):
                    if_tracking_keyframe = True
                    self.from_last_tracking_keyframe = 0
                else:
                    if_tracking_keyframe = False
                    self.from_last_tracking_keyframe += 1
                
                # Mapping keyframe
                if (self.from_last_tracking_keyframe) % self.keyframe_freq == 0:
                    if_mapping_keyframe = True
                else:
                    if_mapping_keyframe = False
                
                if if_tracking_keyframe:                    
                    while self.is_tracking_keyframe_shared[0] or self.is_mapping_keyframe_shared[0]:
                        time.sleep(1e-15)
                    
                    rots = np.array(self.reg.get_source_rotationsq())
                    rots = np.reshape(rots, (-1,4))

                    R_d = Rotation.from_matrix(R)    # from camera R
                    R_d_q = R_d.as_quat()            # xyzw
                    rots = self.quaternion_multiply(R_d_q, rots)
                    
                    scales = np.array(self.reg.get_source_scales())
                    scales = np.reshape(scales, (-1,3))
                    
                    # Erase overlapped points from current pointcloud before adding to map gaussian #
                    # Using filter
                    not_overlapped_indices_of_trackable_points = self.eliminate_overlapped2(distances, self.overlapped_th2) # 5e-5 self.overlapped_th
                    trackable_filter = trackable_filter[not_overlapped_indices_of_trackable_points]
                    
                    # Add new gaussians
                    self.shared_new_gaussians.input_values(torch.tensor(points), torch.tensor(colors), 
                                                       torch.tensor(rots), torch.tensor(scales), 
                                                       torch.tensor(z_values), torch.tensor(trackable_filter))

                    # Add new keyframe
                    depth_image = depth_image.astype(np.float32)/self.depth_scale
                    self.shared_cam.setup_cam(R, T, current_image, depth_image)
                    self.shared_cam.cam_idx[0] = self.iteration_images
                    
                    self.is_tracking_keyframe_shared[0] = 1
                    
                    # Get new target point
                    while not self.target_gaussians_ready[0]:
                        time.sleep(1e-15)
                    target_points, target_rots, target_scales = self.shared_target_gaussians.get_values_np()
                    self.reg.set_input_target(target_points)
                    self.reg.set_target_covariances_fromqs(target_rots.flatten(), target_scales.flatten())
                    self.target_gaussians_ready[0] = 0

                elif if_mapping_keyframe:                    
                    while self.is_tracking_keyframe_shared[0] or self.is_mapping_keyframe_shared[0]:
                        time.sleep(1e-15)
                    
                    rots = np.array(self.reg.get_source_rotationsq())
                    rots = np.reshape(rots, (-1,4))

                    R_d = Rotation.from_matrix(R)    # from camera R
                    R_d_q = R_d.as_quat()            # xyzw
                    rots = self.quaternion_multiply(R_d_q, rots)
                                        
                    scales = np.array(self.reg.get_source_scales())
                    scales = np.reshape(scales, (-1,3))

                    self.shared_new_gaussians.input_values(torch.tensor(points), torch.tensor(colors), 
                                                       torch.tensor(rots), torch.tensor(scales), 
                                                       torch.tensor(z_values), torch.tensor(trackable_filter))
                    
                    # Add new keyframe
                    depth_image = depth_image.astype(np.float32)/self.depth_scale
                    self.shared_cam.setup_cam(R, T, current_image, depth_image)
                    self.shared_cam.cam_idx[0] = self.iteration_images
                    
                    self.is_mapping_keyframe_shared[0] = 1
            pbar.update(1)
                
            self.iteration_images += 1
        
        # Tracking end
        pbar.close()
        self.final_pose[:,:,:] = torch.tensor(self.poses).float()
        self.end_of_dataset[0] = 1
        
        print(f"System FPS: {1/((time.time()-self.total_start_time)/self.num_images):.2f}")
        print(f"ATE RMSE: {self.evaluate_ate(self.trajmanager.gt_poses, self.poses)*100.:.2f}")

    
    def get_images(self, images_folder):
        rgb_images = []
        depth_images = []
        if self.trajmanager.which_dataset == "replica":
            image_files = os.listdir(images_folder)
            image_files = sorted(image_files.copy())
            for key in tqdm(image_files): 
                image_name = key.split(".")[0]
                depth_image_name = f"depth{image_name[5:]}"
                
                rgb_image = cv2.imread(f"{self.dataset_path}/images/{image_name}.jpg")
                depth_image = np.array(o3d.io.read_image(f"{self.dataset_path}/depth_images/{depth_image_name}.png"))
                
                rgb_images.append(rgb_image)
                depth_images.append(depth_image)
            return rgb_images, depth_images
        elif self.trajmanager.which_dataset == "tum":
            for i in tqdm(range(len(self.trajmanager.color_paths))):
                rgb_image = cv2.imread(self.trajmanager.color_paths[i])
                depth_image = np.array(o3d.io.read_image(self.trajmanager.depth_paths[i]))
                rgb_images.append(rgb_image)
                depth_images.append(depth_image)
            return rgb_images, depth_images

    def run_viewer(self, lower_speed=True):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            if time.time()-self.last_t < 1/self.viewer_fps and lower_speed:
                break
            try:
                net_image_bytes = None
                custom_cam, do_training, self.pipe.convert_SHs_python, self.pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    
                    # net_image = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer)["render_depth"]
                    # net_image = torch.concat([net_image,net_image,net_image], dim=0)
                    # net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=7.0) * 50).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    
                self.last_t = time.time()
                network_gui.send(net_image_bytes, self.dataset_path) 
                if do_training and (not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

    def quaternion_multiply(self, q1, Q2):
        # q1*Q2
        x0, y0, z0, w0 = q1
        
        return np.array([w0*Q2[:,0] + x0*Q2[:,3] + y0*Q2[:,2] - z0*Q2[:,1],
                        w0*Q2[:,1] + y0*Q2[:,3] + z0*Q2[:,0] - x0*Q2[:,2],
                        w0*Q2[:,2] + z0*Q2[:,3] + x0*Q2[:,1] - y0*Q2[:,0],
                        w0*Q2[:,3] - x0*Q2[:,0] - y0*Q2[:,1] - z0*Q2[:,2]]).T

    def set_downsample_filter( self, downsample_scale):
        # Get sampling idxs
        sample_interval = downsample_scale
        h_val = sample_interval * torch.arange(0,int(self.H/sample_interval)+1)
        h_val = h_val-1
        h_val[0] = 0
        h_val = h_val*self.W
        a, b = torch.meshgrid(h_val, torch.arange(0,self.W,sample_interval))
        # For tensor indexing, we need tuple
        pick_idxs = ((a+b).flatten(),)
        # Get u, v values
        v, u = torch.meshgrid(torch.arange(0,self.H), torch.arange(0,self.W))
        u = u.flatten()[pick_idxs]
        v = v.flatten()[pick_idxs]
        
        # Calculate xy values, not multiplied with z_values
        x_pre = (u-self.cx)/self.fx # * z_values
        y_pre = (v-self.cy)/self.fy # * z_values
        
        return pick_idxs, x_pre, y_pre

    def downsample_and_make_pointcloud2(self, depth_img, rgb_img):
        
        colors = torch.from_numpy(rgb_img).reshape(-1,3).float()[self.downsample_idxs]/255
        z_values = torch.from_numpy(depth_img.astype(np.float32)).flatten()[self.downsample_idxs]/self.depth_scale
        zero_filter = torch.where(z_values!=0)
        filter = torch.where(z_values[zero_filter]<=self.depth_trunc)
        # print(z_values[filter].min())
        # Trackable gaussians (will be used in tracking)
        z_values = z_values[zero_filter]
        x = self.x_pre[zero_filter] * z_values
        y = self.y_pre[zero_filter] * z_values
        points = torch.stack([x,y,z_values], dim=-1)
        colors = colors[zero_filter]
        
        # untrackable gaussians (won't be used in tracking, but will be used in 3DGS)
        
        return points.numpy(), colors.numpy(), z_values.numpy(), filter[0].numpy()
    
    def eliminate_overlapped2(self, distances, threshold):
        
        # plt.hist(distances, bins=np.arange(0.,0.003,0.00001))
        # plt.show()
        new_p_indices = np.where(distances>threshold)    # 5e-5
        
        return new_p_indices
        
    def align(self, model, data):

        np.set_printoptions(precision=3, suppress=True)
        model_zerocentered = model - model.mean(1).reshape((3,-1))
        data_zerocentered = data - data.mean(1).reshape((3,-1))

        W = np.zeros((3, 3))
        for column in range(model.shape[1]):
            W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
        U, d, Vh = np.linalg.linalg.svd(W.transpose())
        S = np.matrix(np.identity(3))
        if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
            S[2, 2] = -1
        rot = U*S*Vh
        trans = data.mean(1).reshape((3,-1)) - rot * model.mean(1).reshape((3,-1))

        model_aligned = rot * model + trans
        alignment_error = model_aligned - data

        trans_error = np.sqrt(np.sum(np.multiply(
            alignment_error, alignment_error), 0)).A[0]

        return rot, trans, trans_error

    def evaluate_ate(self, gt_traj, est_traj):

        gt_traj_pts = [gt_traj[idx][:3,3] for idx in range(len(gt_traj))]
        gt_traj_pts_arr = np.array(gt_traj_pts)
        gt_traj_pts_tensor = torch.tensor(gt_traj_pts_arr)
        gt_traj_pts = torch.stack(tuple(gt_traj_pts_tensor)).detach().cpu().numpy().T

        est_traj_pts = [est_traj[idx][:3,3] for idx in range(len(est_traj))]
        est_traj_pts_arr = np.array(est_traj_pts)
        est_traj_pts_tensor = torch.tensor(est_traj_pts_arr)
        est_traj_pts = torch.stack(tuple(est_traj_pts_tensor)).detach().cpu().numpy().T

        _, _, trans_error = self.align(gt_traj_pts, est_traj_pts)

        avg_trans_error = trans_error.mean()

        return avg_trans_error