# RGBD GS-ICP SLAM

Seongbo Ha, Jiung Yeon, Hyeonwoo Yu

Paper | Video

This repository is intended to substantiate the results reported in the paper. Additional features including visualization tools will be updated soon!

## Environments
Install requirements
```bash
conda create -n gsicpslam python==3.9
conda activate gsicpslam
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```
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