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

## Prepare input data
cd /home/ubuntu && mkdir input





