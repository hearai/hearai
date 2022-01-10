# hearai/pose_estimation
Here we are going to check different pose estimation libraries:
1. requirements and functionality
2. performance
3. comparison

https://3.basecamp.com/3105098/buckets/24328381/todolists/4382490071

# Openpose build tutorial

## Create the container
cd hearai\pose_estimation\open_pose && ./start.sh

## Enter the container
./enter.sh

## Prepare build configuration
ccmake .
Insert console build setup - in this section, in console you have to disable -DUSE_CUDNN=OFF and enable -DUSE_PYTHON=ON, then press c to configure

## Openpose build
cd openpose/build && make -j`nproc`
## Download models
cd ../models/ && ./getModels.sh

## Prepare input/output data directories
cd /home/ubuntu && mkdir input
mkdir output

# Work with openpose

## Prepare data

In order to start working, you need to upload the video to be processed in the container. I recommend to use 2 consoles - 1st for host and 2nd for container.
 If you want to work with only one console, press Ctrl-P, followed by Ctrl-Q, to detach from your connection and use: 
 cd hearai\pose_estimation\open_pose && ./enter.sh
to bring it back.

[Host Console]
1. First you have to find your container id - the name of your container should be hearai_{user_id}):
docker ps

2. Next you need to upload video to the container:
docker cp Host/Path/video.avi container id:/home/ubuntu/input

## Data processing

[Container console]
1. Go to the openpose path:
cd openpose

2. Execute command where video.avi = your video
./build/examples/openpose/openpose.bin --video ../input/video.avi --hand --write_video ../output/video.avi --display 0 --number_people_max 1 --net_resolution -1x128

3. If you wanna try more flags please check website:
https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/01_demo.md

## Read data

[Host Console]
1. To check how it works, you need to copy the processed video to the host directory:
docker cp container id:/home/ubuntu/output/video.avi /Host/Path/