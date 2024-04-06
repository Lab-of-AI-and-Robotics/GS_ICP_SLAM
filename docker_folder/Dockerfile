FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y upgrade
RUN apt-get -y install sudo
RUN adduser --disabled-password --gecos "" user  \
    && echo 'user:user' | chpasswd \
    && adduser user sudo \
    && echo 'user ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# install python3.9
RUN sudo apt-get update
RUN sudo apt-get -y upgrade
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y keyboard-configuration
RUN sudo DEBIAN_FRONTEND=noninteractive apt install -y python3.9 python3.9-dev
RUN sudo DEBIAN_FRONTEND=noninteractive apt-get -y install wget python3-pip build-essential git curl lsb-release

# symlink
RUN sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN ln -s -f /usr/bin/python /usr/bin/python3

# For pcl (including pcl visualizer)
RUN sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
RUN sudo apt update
RUN sudo apt install -y ros-noetic-desktop-full

# GS-ICP SLAM
WORKDIR /home
RUN git clone --recursive https://github.com/Lab-of-AI-and-Robotics/GS_ICP_SLAM.git

## Dependency
RUN echo "export CUDA_HOME=/usr/local/cuda && export PATH=/usr/local/cuda-11.8/bin:$PATH && LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
RUN pip install opencv-contrib-python scipy tqdm plyfile
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install torchmetrics
RUN sudo apt-get install -y ninja-build
RUN pip3 install --upgrade pip
RUN pip3.9 install open3d --ignore-installed PyYAML

## fast_gicp
WORKDIR /home/GS_ICP_SLAM
WORKDIR /home/GS_ICP_SLAM/submodules/fast_gicp
RUN mkdir build
WORKDIR /home/GS_ICP_SLAM/submodules/fast_gicp/build
RUN cmake ..
RUN make -j4
WORKDIR /home/GS_ICP_SLAM/submodules/fast_gicp
RUN python setup.py install --user

# RUN pip install submodules/diff-gaussian-rasterization
# RUN pip install submodules/simple-knn

## Viewer
### Dependency
RUN sudo apt install -y libglew-dev libgtk-3-dev libassimp-dev libglfw3-dev libavdevice-dev libxxf86vm-dev
#### opencv (ver 4.5 required)
RUN sudo apt-get install -y libjpeg-dev libtiff5-dev libpng-dev \
                        libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev \
                        libv4l-dev v4l-utils \
                        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
                        libgtk2.0-dev mesa-utils libgl1-mesa-dri libgtkgl2.0-dev libgtkglext1-dev \
                        libatlas-base-dev gfortran libeigen3-dev
WORKDIR /home
RUN mkdir opencv
WORKDIR /home/opencv
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.5.zip
RUN wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.5.5.zip
RUN unzip opencv.zip && unzip opencv_contrib.zip
RUN rm opencv.zip && rm opencv_contrib.zip
WORKDIR /home/opencv/opencv-4.5.5
RUN mkdir build
WORKDIR /home/opencv/opencv-4.5.5/build
RUN cmake -D OPENCV_EXTRA_MODULES_PATH=/home/opencv/opencv_contrib-4.5.5/modules ..
RUN make -j4
RUN sudo make install

#### embree 3.0
WORKDIR /home
RUN wget https://github.com/embree/embree/archive/refs/tags/v3.0.0.tar.gz
RUN tar -zxvf v3.0.0.tar.gz
WORKDIR /home/embree-3.0.0
RUN mkdir build
WORKDIR /home/embree-3.0.0/build
RUN cmake -D EMBREE_ISPC_SUPPORT=False ..
RUN make -j4
RUN sudo make install

### build / install
WORKDIR /home/GS_ICP_SLAM/SIBR_viewers
RUN cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build -j4 --target install