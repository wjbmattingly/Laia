### Input arguments (https://github.com/moby/moby/pull/31352) ###
#ARG CUDA=8.0

### Start with CUDA Torch ###
FROM cuda-torch:8.0

MAINTAINER Mauricio Villegas <mauricio_ville@yahoo.com>

### Install pre-requisits ###
RUN apt-get update --fix-missing \
 && apt-get install -y --no-install-recommends \
      libgoogle-glog-dev \
      libgtest-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

### Install warp-ctc and imgdistort ###
RUN luarocks install http://raw.githubusercontent.com/baidu-research/warp-ctc/master/torch_binding/rocks/warp-ctc-scm-1.rockspec \
 && luarocks install http://raw.githubusercontent.com/jpuigcerver/imgdistort/master/torch/imgdistort-scm-1.rockspec

#RUN luarocks install http://raw.githubusercontent.com/baidu-research/warp-ctc/master/torch_binding/rocks/warp-ctc-scm-1.rockspec
#RUN git clone https://github.com/jpuigcerver/imgdistort.git /tmp/imgdistort \
# && sed 's|VERSION 3.0|VERSION 2.8|' -i /tmp/imgdistort/CMakeLists.txt \
# && luarocks make

### Install Laia ###
RUN LIBRARY_PATH=$(echo "$LIBRARY_PATH" | sed 's|^:||; s|:$||; s|::|:|g;') \
 && luarocks install http://raw.githubusercontent.com/jpuigcerver/Laia/master/rocks/laia-scm-1.rockspec

### Remove temporal files ###
RUN rm -fr /tmp/* \
 && cd ~/torch \
 && sed 's| -o -name install||' -i clean.sh \
 && ./clean.sh
