# Docker Setup Guide

This Dockerfile contains all requirements for GS-ICP SLAM and SIBR_viewer.

## Prerequisites
- Docker and nvidia-docker2 must be installed

## Step 1: Clone and Build Docker Image
```bash
git clone --recursive https://github.com/Lab-of-AI-and-Robotics/GS_ICP_SLAM.git
cd GS_ICP_SLAM/
cd docker_folder
docker build -t gsidocker:latest .
```

## Step 2: Download Datasets (Host Environment)
Download Replica dataset:
```bash
cd GS_ICP_SLAM/
bash download_replica.sh 
```

Download TUM dataset:
```bash
cd GS_ICP_SLAM/
bash download_tum.sh
```

## Step 3: Reorganize Replica Dataset Structure
Create and run the reorganization script:
```bash
gedit ~/reorganize_replica.sh
chmod +x ~/reorganize_replica.sh
bash ~/reorganize_replica.sh
```

**reorganize_replica.sh content:**
```bash
#!/bin/bash

# Replica dataset path
REPLICA_PATH="$HOME/dataset/Replica"

# Process all directories (room0, room1, office0, office1, etc.)
for dir in $REPLICA_PATH/*/; do
    # Skip mesh files
    if [[ "$dir" == *"_mesh.ply" ]]; then
        continue
    fi
    
    # Extract directory name
    dirname=$(basename "$dir")
    
    # Process only room* or office* directories
    if [[ "$dirname" == room* ]] || [[ "$dirname" == office* ]]; then
        echo "Processing: $dirname"
        
        # Check if results folder exists
        if [ -d "$dir/results" ]; then
            cd "$dir"
            
            # Create new folders
            mkdir -p images depth_images
            
            # Move RGB images
            if ls results/frame*.jpg 1> /dev/null 2>&1; then
                mv results/frame*.jpg images/
                echo "  - Moved RGB images to images/"
            fi
            
            # Move depth images (png format)
            if ls results/depth*.png 1> /dev/null 2>&1; then
                mv results/depth*.png depth_images/
                echo "  - Moved depth images to depth_images/"
            fi
            
            # Move depth images (jpg format)
            if ls results/depth*.jpg 1> /dev/null 2>&1; then
                mv results/depth*.jpg depth_images/
                echo "  - Moved depth images (jpg) to depth_images/"
            fi
            
            # Remove results folder if empty
            if [ -z "$(ls -A results)" ]; then
                rmdir results
                echo "  - Removed empty results folder"
            else
                echo "  - Warning: results folder not empty"
                ls results/
            fi
        else
            echo "  - No results folder found, skipping"
        fi
        echo ""
    fi
done

echo "Reorganization complete!"
```

## Step 4: Create Docker Container

### Container Configuration
When creating the Docker container, you must configure:
- **Dataset directory mapping**: Links host dataset directory to container
  - `-v {host_dataset_path}:/home/dataset`
- **Shared memory size**: Default 4MB is insufficient, recommended 12GB
  - `--shm-size 12gb`

### Initial Container Creation
```bash
docker run -it \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  -e USER=$USER \
  -e runtime=nvidia \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -v /home/hyeon/GS_ICP_SLAM:/home/GS_ICP_SLAM \
  -v /home/hyeon/dataset:/home/dataset \
  --shm-size 12gb \
  --net host \
  --gpus all \
  --privileged \
  --name gsicpslam \
  gsidocker:latest \
  /bin/bash
```

### Restart Existing Container
```bash
xhost +local:docker
docker start gsicpslam
docker exec -it gsicpslam bash
```

## Step 5: Install Dependencies (Inside Container)

The fast_gicp submodule may already be installed during Docker image build.
Install remaining submodules and dependencies:

```bash
cd /home/GS_ICP_SLAM
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

cd submodules/fast_gicp
mkdir build
cd build
cmake ..
make
cd ..
python setup.py install

pip install rerun-sdk

# Downgrade NumPy to 1.x version for compatibility
pip uninstall numpy -y
pip install numpy==1.24.3

# Install additional packages
pip install torchmetrics

# Verify NumPy version
python -c "import numpy; print(numpy.__version__)"
```

## Step 6: Run the Project
```bash
bash replica.sh
```

## Notes
- **Dataset Path**: Adjust `/home/hyeon/` paths to match your environment
- **GPU Support**: Requires NVIDIA GPU with nvidia-docker2
- **Memory**: Minimum 12GB shared memory recommended
- **X11 Forwarding**: Configured for GUI applications
- **Dataset Directory**: The system expects datasets in `/home/dataset` inside the container, which is mapped from your host dataset directory