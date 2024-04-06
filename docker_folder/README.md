# Docker
Our Dockerfile contains requirements for either GS-ICP SLAM and SIBR_viewer.

## Requirements
Docker and nvidia-docker2 must be installed.

## Make docker image from Dockerfile
```bash
cd docker_folder
docker build -t gsidocker:latest .
```

## Make GS-ICP SLAM container

When making docker container, users must set 'dataset directory of main environment' and 'shared memory size'.
- Dataset directory of main environment
  - We can link a directory of main environment and that of docker container. So without downloading Replica/TUM dataset in the docker container, we can use datasets downloaded in the main environment.
  - -v {dataset directory of main environment}:{dataset directory of docker container}
- shared memory size
  - The size of shared memory is 4mb in default, and this is not sufficient for our system. When testing, I set this value as 12G and it worked well.

An example command for making a container is shown below.
```bash
# In main environment
docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -e USER=$USER \
-e runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_VISIBLE_DEVICES=all \
-v {dataset directory of main environment}:/home/dataset --shm-size {shared memory size} \
--net host --gpus all --privileged --name gsicpslam gsidocker:latest /bin/bash
```

## Install submodules
The fast_gicp submodule may already installed while making docker image from Dockerfile.
So users need to install only 'diff-gaussian-rasterization' and 'simple_knn' submodules.
```bash
# In docker container
cd /home/GS_ICP_SLAM
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple_knn
```

## Edit dataset directory

In our system, the directory of datasets are defined as GS_ICP_SLAM/dataset, so we need to change it to the dataset directory in the docker container.
ex)
replica.sh
```bash
OUTPUT_PATH="experiments/results"
DATASET_PATH="dataset/Replica" #<- Change this to the dataset directory in the docker container
```