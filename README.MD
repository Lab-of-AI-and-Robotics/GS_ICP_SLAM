<div align=center>

# RGBD GS-ICP SLAM (100FPS Gaussian Splatting SLAM)

[Seongbo Ha](https://riboha.github.io), [Jiung Yeon](https://humdrum-balance-b8f.notion.site/Jiung-Yeon-6754922a22814c9a95af88801a96fb4b), Hyeonwoo Yu

<h3 align="center"> ECCV 2024 </h3>

[Paper](https://arxiv.org/abs/2403.12550) | [Video](https://www.youtube.com/watch?v=e-bHh_uMMxE&t)

![github (1)](https://github.com/Lab-of-AI-and-Robotics/GS_ICP_SLAM/assets/34827206/5722e8f4-165d-4093-8064-a7ed5d9ea008)

</div>

This repository is intended to substantiate the results reported in the paper. Additional features including visualization tools will be updated soon!

## Environments
Install requirements
```bash
conda create -n gsicpslam python==3.9
conda activate gsicpslam
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```
Also, PCL is needed for fast-gicp submodule.

Install submodules

```bash
conda activate gsicpslam
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

cd submodules/fast_gicp
mkdir build
cd build
cmake ..
make
cd ..
python setup.py install --user
```


## Datasets

- Replica
  - Download
    ```bash
    bash download_replica.sh
    ```
  - Configure
  
    Please modify the directory structure to ours.

    The original structure
    ```bash
    Replica
        - room0
            - results (contain rgbd images)
                - frame000000.jpg
                - depth000000.jpg
                ...
            - traj.txt
        ...
    ```
    Our structure
    ```bash
    Replica
        - room0
            - images (contain rgb images)
                - frame000000.jpg
                ...
            - depth_images (contain depth images)
                - depth000000.jpg
                ...
            - traj.txt
        ...
    ```    

- TUM-RGBD
  - Download
    ```bash
        bash download_tum.sh
    ```

## Run
- Limited to 30 FPS
    ```bash
        # Replica
        bash replica.sh
        # TUM
        bash tum.sh
    ```

- Unlimited tracking speed
    ```bash
        # Replica
        bash replica_unlimit.sh
        # TUM
        bash tum_unlimit.sh
    ```

## Installing SIBR Viewer
```bash
cd SIBR_viewers
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j24 --target install
```

## Real-time demo
### Using rerun.io viewer

Rerun viewer shows the means of trackable Gaussians, and rendered image from reconstructed 3dgs map.

![GIFMaker_me](https://github.com/Lab-of-AI-and-Robotics/GS_ICP_SLAM/assets/34827206/b4715071-2e4a-4d17-b7a2-612bbd32dbd0)

```bash
python -W ignore gs_icp_slam.py --rerun_viewer
```


### Using SIBR viewer
```bash
python -W ignore gs_icp_slam.py --dataset_path dataset/Replica/office0 --verbose

# In other terminal
cd SIBR_viewers
./install/bin/SIBR_remoteGaussian_app --rendering-size 1280 720
```

## Docker
Please see the README.md in the docker_files folder.
