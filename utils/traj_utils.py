import os
import sys
import numpy as np
from matplotlib import pyplot
from scipy.spatial.transform import Rotation as R

class TrajManager:
    def __init__(self, which_dataset, dataset_path):
        '''
        For plot/evaluate trajectory
        
        Args:
        which_dataset : "tum" or "replica"
        dataset_path : dataset path
        '''
        self.which_dataset = which_dataset
        self.dataset_path = dataset_path

        if self.which_dataset == "tum":
            self.gt_poses = self.tum_load_poses(self.dataset_path + '/traj.txt')
        elif self.which_dataset == "replica":
            self.gt_poses = self.replica_load_poses(self.dataset_path + '/traj.txt')
        else:
            print("Unknown dataset!")
            sys.exit()
        
        self.gt_poses_vis = np.array([x[:3, 3] for x in self.gt_poses])

    def quaternion_rotation_matrix(self, Q, t):
        r = R.from_quat(Q)
        rotation_mat = r.as_matrix()
        # rotation_mat = np.transpose(rotation_mat)
        # rotation_mat = np.linalg.inv(rotation_mat)
        T = np.empty((4, 4))
        T[:3, :3] = rotation_mat
        T[:3, 3] = [t[0], t[1], t[2]]

        T[3, :] = [0, 0, 0, 1]     
        # return np.linalg.inv(T)
        return T
    
    def replica_load_poses(self, path):
        poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            # c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        return np.array(poses)

    def pose_matrix_from_quaternion(self, pvec):
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

    def tum_load_poses(self, path):
        # poses = []
        # with open(path, "r") as f:
        #     lines = f.readlines()
        # for i in range(len(lines)):
        #     line = lines[i].split()
        #     xyz = np.array([  	float(line[1]),
        #                                 float(line[2]),
        #                                 float(line[3])])
        #     q = np.array([	float(line[4]),
        #                     float(line[5]),
        #                     float(line[6]),
        #                     float(line[7])])
        #     c2w = self.quaternion_rotation_matrix(q, xyz)
        #     poses.append(c2w)
        frame_rate = 32

        if os.path.isfile(os.path.join(self.dataset_path, 'groundtruth.txt')):
            pose_list = os.path.join(self.dataset_path, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(self.dataset_path, 'pose.txt')):
            pose_list = os.path.join(self.dataset_path, 'pose.txt')

        image_list = os.path.join(self.dataset_path, 'rgb.txt')
        depth_list = os.path.join(self.dataset_path, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        self.color_paths, poses, self.depth_paths, = [], [], []
        for ix in indicies:
            (i, j, k) = associations[ix]
            self.color_paths += [os.path.join(self.dataset_path, image_data[i, 1])]
            self.depth_paths += [os.path.join(self.dataset_path, depth_data[j, 1])]
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            poses += [c2w]
        
        return np.array(poses)
    
    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.unicode_, skiprows=skiprows)
        return data
    
    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations
    
    def plot_traj(self, iter, poses):
        '''
        Plot trajectory
        
        iter : iter
        poses : list of estimated poses
        '''
        iter += 1
        traj = np.array([x[:3, 3] for x in poses])
        pyplot.clf()
        pyplot.title(f'Seq {iter}')
        pyplot.plot(traj[:, 0], traj[:, 1], label='estimated trajectory', linewidth=3)
        pyplot.legend()
        pyplot.plot(self.gt_poses_vis[:iter, 0], self.gt_poses_vis[:iter, 1], label='g.t. trajectory')
        pyplot.legend()
        pyplot.axis('equal')
        pyplot.pause(1e-15)
    
    def eval_traj(self):
        pass

if __name__ =="__main__":
    a = TrajManager("scannetpp", "/home/kdg/GS_ICP_SLAM/src/dataset/Scannetpp/8b5caf3398/transforms_undistorted.json")