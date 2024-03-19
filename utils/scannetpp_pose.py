import os
import sys
import numpy as np
from matplotlib import pyplot
from scipy.spatial.transform import Rotation as R
import json
from tqdm import tqdm

# image_dir = "/home/kdg/GS_ICP_SLAM/src/dataset/Scannetpp/8b5caf3398/images"
# output_dir = "/home/kdg/GS_ICP_SLAM/src/dataset/Scannetpp/8b5caf3398/traj.txt"

image_dir = "/home/kdg/GS_ICP_SLAM/src/dataset/Scannetpp/b20a261fdf/images"
output_dir = "/home/kdg/GS_ICP_SLAM/src/dataset/Scannetpp/b20a261fdf/traj.txt"

txt_file = open(output_dir, "w")

def create_filepath_index_mapping(frames):
    return {frame["file_path"]: index for index, frame in enumerate(frames)}

# cams_metadata = json.load(open("/home/kdg/GS_ICP_SLAM/src/dataset/Scannetpp/8b5caf3398/transforms_undistorted.json", "r"))
cams_metadata = json.load(open("/home/kdg/GS_ICP_SLAM/src/dataset/Scannetpp/b20a261fdf/transforms_undistorted.json", "r"))

filepath_index_mapping = create_filepath_index_mapping(cams_metadata["frames"])

image_files = os.listdir(image_dir)
image_files = sorted(image_files.copy())

for key in tqdm(image_files):
    matched = cams_metadata["frames"][filepath_index_mapping.get(key)]
    pose = np.array(matched["transform_matrix"]).flatten()
    txt_file.write(f"{str(pose[0])} {str(pose[1])} {str(pose[2])} {str(pose[3])} {str(pose[4])} {str(pose[5])} {str(pose[6])} {str(pose[7])} {str(pose[8])} {str(pose[9])} {str(pose[10])} {str(pose[11])} {str(pose[12])} {str(pose[13])} {str(pose[14])} {str(pose[15])}\n")
    print(f"image name/matched traj : {key}/{matched['file_path']}\n")

txt_file.close()
    