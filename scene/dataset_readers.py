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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import cv2
import re

def read_depth_normalized(path, normalized_const=1.):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    
    return Image.fromarray(np.transpose(array, (1, 0, 2)).squeeze()/normalized_const)

def read_depth_png(path, normalized_const=1.):
    depth_np = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    depth_data = depth_np.astype(np.float32) / float(normalized_const)
    # print(f"\ndepth min/max {depth_data.min()}/{depth_data.max()}") # ok
    return Image.fromarray(depth_data)
 
def caminfos2pcd(cam_infos, sampling_ratio=0.2):
    points = []
    colors = []
    cam_num = len(cam_infos)
    sampling_ratio *= 1./cam_num
    for idx, cam_info in enumerate(cam_infos):
        FocalX = cam_info.FocalX
        FocalY = cam_info.FocalY
        CenterX = cam_info.CenterX
        CenterY = cam_info.CenterY
        R = cam_info.R
        T = cam_info.T
        image = cam_info.image
        depth_image = cam_info.depth_image
        if depth_image is not None:
            u_list = np.tile([i for i in range(cam_info.width)], cam_info.height).reshape(cam_info.height, cam_info.width)
            v_list = np.tile([j for j in range(cam_info.height)], cam_info.width).reshape(cam_info.width, cam_info.height).transpose()
            x = (u_list - CenterX) * depth_image / FocalX
            y = (v_list - CenterY) * depth_image / FocalY
            points_curr = np.stack([x,y,depth_image], axis=-1).reshape([-1,3])
            # i don't know why, but in colmap format it seems like it uses R^t, T as the projection from world to the camera
            # thus in Gaussian Platting view2world matrix is calculated by using: [R, -R*T] 
            # (see line 38 in gaussian-splatting/utils/graphics_utils.py and line 54 in gaussian-splatting/scene/cameras.py
            points_curr = np.matmul(R, points_curr.transpose()).transpose() - np.matmul(R, T)
            colors_curr = np.array(image).reshape([-1,3])
            points.append(points_curr)
            colors.append(colors_curr)    
    if len(points) == 0:
        return None, None
    else:
        points = np.concatenate(points, axis=0)
        colors = np.concatenate(colors, axis=0)
        selected_indices = np.random.choice(len(points), size=(int)(len(points)*sampling_ratio), replace=False)
        points = points[selected_indices]
        colors = colors[selected_indices]
        return points, colors 
        

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    FocalX: np.array
    FocalY: np.array
    CenterX: np.array
    CenterY: np.array
    image: np.array
    depth_image: np.array
    image_path: str
    image_name: str
    depth_image_name : str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, depth_images_folder=None):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            FocalX = intr.params[0]
            FocalY = intr.params[0]            
            FovY = focal2fov(FocalY, height)
            FovX = focal2fov(FocalX, width)
            CenterX = intr.params[1]
            CenterY = intr.params[2]
        elif intr.model=="PINHOLE":
            FocalX = intr.params[0]
            FocalY = intr.params[1]
            FovY = focal2fov(FocalY, height)
            FovX = focal2fov(FocalX, width)
            CenterX = intr.params[2]
            CenterY = intr.params[3]
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        
        try:
            depth_image_path = os.path.join(depth_images_folder, image_name + '.bin')
            depth_image = read_depth_normalized(depth_image_path)
        except:
            depth_image = None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, FocalX=FocalX, FocalY=FocalY, CenterX=CenterX, CenterY=CenterY, 
                              image=image, depth_image=depth_image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readSLAMCameras(cam_intrinsics, images_folder, depth_images_folder=None):
    cam_infos = []
    image_files = os.listdir(images_folder)
    which_dataset = cam_intrinsics[8]    # replica/tum/scannet
    
    for idx, key in enumerate(image_files):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(image_files)))
        sys.stdout.flush()

        R = np.identity(3)
        T = np.array([0,0,0])
        width = int(cam_intrinsics[0])
        height = int(cam_intrinsics[1])
        FocalX = float(cam_intrinsics[2])
        FocalY = float(cam_intrinsics[3])
        FovY = focal2fov(FocalY, height)
        FovX = focal2fov(FocalX, width)
        CenterX = float(cam_intrinsics[4])
        CenterY = float(cam_intrinsics[5])
        uid = idx
        
        image_path = os.path.join(images_folder, key)
        image_name = key.split(".")[0]
        
        # Too many open files 수정
        temp = Image.open(image_path)
        image = temp.copy()
        
        if which_dataset=="replica":
            depth_image_path = os.path.join(depth_images_folder, f"depth{image_name[5:]}.png")
            depth_image_name = f"depth{image_name[5:]}"
            temp_depth = read_depth_png(depth_image_path, cam_intrinsics[6])
            # print(np.array(temp_depth)) ok
            depth_image = temp_depth.copy()
        elif which_dataset=="tum":
            depth_image_path = os.path.join(depth_images_folder, f"{image_name}.png")
            depth_image_name = image_name
            temp_depth = read_depth_png(depth_image_path, cam_intrinsics[6])
            depth_image = temp_depth.copy()

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, FocalX=FocalX, FocalY=FocalY, CenterX=CenterX, CenterY=CenterY, 
                              image=image, depth_image=depth_image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              depth_image_name = depth_image_name)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), depth_images_folder=os.path.join(path, "depth_images"))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        xyz, rgb = caminfos2pcd(cam_infos)
        if xyz is None:
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readSLAMSceneInfo(path, images, eval, llffhold=8):


    reading_dir = "images" if images == None else images
    # Read camera intrinsics
    cam_intrinsic_file = open(f"{path}/caminfo.txt")
    cam_intrinsics_ = cam_intrinsic_file.readlines()
    cam_intrinsics = cam_intrinsics_[2].split()
    
    cam_infos_unsorted = readSLAMCameras(cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), depth_images_folder=os.path.join(path, "depth_images"))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)


    train_cam_infos = cam_infos
    test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    pcd = None
    ply_path = None
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "SLAM" : readSLAMSceneInfo
}
