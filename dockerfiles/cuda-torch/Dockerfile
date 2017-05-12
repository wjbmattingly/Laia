### Input arguments (https://github.com/moby/moby/pull/31352) ###
#ARG CUDA=8.0

### Start with CUDA Torch dependencies ###
FROM cuda-torch-deps2:8.0

MAINTAINER Mauricio Villegas <mauricio_ville@yahoo.com>

### Restore Torch7 installation script ###
#RUN cd ~/torch \
# && sed -i 's/path_to_nvcc=$(which no_nvcc)/path_to_nvcc=$(which nvcc)/g' install.sh

### Install CUDA libraries ###
RUN luarocks install torch \
 && luarocks install cutorch \
 && luarocks install cunn \
 && luarocks install cudnn

### Remove temporal files ###
RUN rm -fr /tmp/*
