### Input arguments (https://github.com/moby/moby/pull/31352) ###
#ARG CUDA=8.0

### Start with CUDA Torch dependencies ###
FROM cuda-torch-deps1:8.0

MAINTAINER Mauricio Villegas <mauricio_ville@yahoo.com>

### Install Torch7 without nvcc to prevent timeouts ###
RUN cd ~/torch \
# && sed -i 's/path_to_nvcc=$(which nvcc)/path_to_nvcc=$(which no_nvcc)/g' install.sh \
# && sed -i 's,path_to_nvcc=/usr/local/cuda/bin/nvcc,path_to_nvcc=,g' install.sh \
 && export PREFIX=/opt/torch \
 && mkdir -p $PREFIX \
 && ./install.sh -b

### Remove temporal files ###
RUN rm -fr /tmp/*

### Export environment variables manually ###
ENV LUA_PATH='/opt/torch/share/lua/5.1/?.lua;/opt/torch/share/lua/5.1/?/init.lua;./?.lua'
ENV LUA_CPATH='/opt/torch/lib/lua/5.1/?.so;/opt/torch/lib/?.so;./?.so'
ENV PATH=/opt/torch/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/torch/lib:$LD_LIBRARY_PATH
ENV DYLD_LIBRARY_PATH=/opt/torch/lib:$DYLD_LIBRARY_PATH
ENV LUA_CPATH='/opt/torch/lib/?.so;'$LUA_CPATH
