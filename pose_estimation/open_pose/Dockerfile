FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04

ENV OPENPOSE_VERSION=1.7
ENV TZ=Europe/Warsaw
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get -y update --fix-missing
# && apt-get -y upgrade
RUN apt-get install -y software-properties-common
RUN apt-get -y update

RUN apt-get install -y sudo wget bzip2 git vim cmake unzip mc nano

RUN apt-get install -y build-essential

# OpenCV
RUN apt-get install -y libopencv-dev
# General dependencies
RUN apt-get install -y libatlas-base-dev libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler
RUN apt-get install -y --no-install-recommends libboost-all-dev
# Remaining dependencies, 14.04
RUN apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev

#RUN apt-get -y update
RUN apt-get install -y python3
RUN apt-get install -y python3-dev
RUN apt-get install -y python3-venv
RUN apt-get install -y python3-distutils
RUN apt-get install -y python3-setuptools
RUN apt-get install -y python3-wheel
RUN apt-get install -y python3-pip

RUN pip3 install --upgrade numpy protobuf opencv-python

# OpenCL Generic
RUN apt-get install -y opencl-headers ocl-icd-opencl-dev
RUN apt-get install -y libviennacl-dev


# Clone Openpose
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose
# Create build directory
RUN cd openpose 
# Add additional dependencies
RUN git submodule update --init --recursive --remote
# Prepraing build config
RUN mkdir build && cd build
RUN sudo apt-get install cmake-curses-gui
# Insert console build setup - in this section, in console, you have to disable -DUSE_CUDNN=OFF
# and enable -DUSE_PYTHON=ON
#RUN ccmake .
# Openpose Building
#RUN cd openpose/build && make -j`nproc`
#Download models
#RUN cd ../models/ && ./getModels.sh

#For full description please read README.md

# settings
ARG user_name=ubuntu
ARG user_id=6012
ARG group_name=ubuntu
ARG group_id=6012

# create user
RUN groupadd -g 6000 data-hearai
RUN groupadd -g ${group_id} ${group_name}
RUN useradd -u ${user_id} -g ${group_id} -G 6000 -d /home/${user_name} \
    --create-home --shell /bin/bash ${user_name}
RUN echo "${user_name} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN chown -R ${user_name}:${group_name} /home/${user_name}

# user settings
ENV HOME /home/${user_name}
ENV LANG en_US.UTF-8

USER ${user_name}

CMD ["/bin/bash"]
