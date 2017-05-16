### Input arguments (https://github.com/moby/moby/pull/31352) ###
#ARG CUDA=8.0

### Start with NVIDIA cuDNN base image ###
FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

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

### Install Torch7 dependencies ###
RUN git clone https://github.com/torch/distro.git ~/torch --recursive \
 && cd ~/torch \
 && bash install-deps
