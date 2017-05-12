### Input arguments (https://github.com/moby/moby/pull/31352) ###
#ARG CUDA=8.0

### Start with CUDA Torch dependencies ###
FROM cuda-torch-deps1:8.0

MAINTAINER Mauricio Villegas <mauricio_ville@yahoo.com>

### Install Torch7 without nvcc to prevent timeouts ###
RUN cd ~/torch \
# && sed -i 's/path_to_nvcc=$(which nvcc)/path_to_nvcc=$(which no_nvcc)/g' install.sh \
# && sed -i 's,path_to_nvcc=/usr/local/cuda/bin/nvcc,path_to_nvcc=,g' install.sh \
 && ./install.sh

# Export environment variables manually
ENV LUA_PATH='/root/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/root/torch/install/share/lua/5.1/?.lua;/root/torch/install/share/lua/5.1/?/init.lua;./?.lua;/root/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua'
#ENV LUA_CPATH='/root/.luarocks/lib/lua/5.1/?.so;/root/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so'
ENV LUA_CPATH='/root/.luarocks/lib/lua/5.1/?.so;/root/torch/install/lib/lua/5.1/?.so;/root/torch/install/lib/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so'
ENV PATH=/root/torch/install/bin:$PATH
ENV LD_LIBRARY_PATH=/root/torch/install/lib:$LD_LIBRARY_PATH
ENV DYLD_LIBRARY_PATH=/root/torch/install/lib:$DYLD_LIBRARY_PATH
ENV LUA_CPATH='/root/torch/install/lib/?.so;'$LUA_CPATH
