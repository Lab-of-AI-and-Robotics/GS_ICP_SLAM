# RGBD GS-ICP SLAM

## Environments
Install requirements
```bash
conda create -n gsicpslam python==3.8.12
conda activate gsicpslam
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