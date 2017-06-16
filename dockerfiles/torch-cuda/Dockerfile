### Input arguments (https://github.com/moby/moby/pull/31352) ###
#ARG CUDA=7.5

### Start with NVIDIA cuDNN base image ###
FROM nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04

MAINTAINER Mauricio Villegas <mauricio_ville@yahoo.com>

### Install pre-requisits ###
RUN apt-get update --fix-missing \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      cmake \
      git \
      libssl-dev \
      sudo \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

### Export environment variables manually ###
ENV LUA_PATH='/opt/torch/share/lua/5.1/?.lua;/opt/torch/share/lua/5.1/?/init.lua;./?.lua'
ENV LUA_CPATH='/opt/torch/lib/lua/5.1/?.so;/opt/torch/lib/?.so;./?.so'
ENV PATH=/opt/torch/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/torch/lib:$LD_LIBRARY_PATH
ENV DYLD_LIBRARY_PATH=/opt/torch/lib:$DYLD_LIBRARY_PATH
ENV LUA_CPATH='/opt/torch/lib/?.so;'$LUA_CPATH

### Install Torch7 dependencies ###
RUN git clone https://github.com/torch/distro.git ~/torch --recursive \
 && cd ~/torch \
 && bash install-deps \

### Install Torch7 ###
 && export PREFIX=/opt/torch \
 && mkdir -p $PREFIX \
 && ./install.sh -b \

### Install CUDA libraries ###
 && luarocks install torch \
 && luarocks install cutorch \
 && luarocks install cunn \
 && luarocks install cudnn \

### Remove temporal files ###
 && rm -fr /tmp/* ~/torch
