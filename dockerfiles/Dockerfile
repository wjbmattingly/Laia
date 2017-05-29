### Input arguments (https://github.com/moby/moby/pull/31352) ###
#ARG CUDA=7.5
#ARG OS=ubuntu14.04

### Start with CUDA Torch ###
FROM cuda-torch:7.5-ubuntu14.04

MAINTAINER Mauricio Villegas <mauricio_ville@yahoo.com>

### Install pre-requisits ###
RUN add-apt-repository -y ppa:george-edison55/cmake-3.x \
 && apt-get update --fix-missing \
 && apt-get purge -y cmake \
 && apt-get install -y --no-install-recommends \
      cmake \
      libgoogle-glog-dev \
      libgtest-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \

### Install warp-ctc and imgdistort ###
 && luarocks install http://raw.githubusercontent.com/baidu-research/warp-ctc/master/torch_binding/rocks/warp-ctc-scm-1.rockspec \
# && luarocks install http://raw.githubusercontent.com/jpuigcerver/imgdistort/master/torch/imgdistort-scm-1.rockspec \

 && git clone https://github.com/jpuigcerver/imgdistort.git /tmp/imgdistort \
 && cd /tmp/imgdistort \
 && luarocks make torch/imgdistort-scm-1.rockspec \
 && cd \
 && rm -fr /tmp/* \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \

### Install Laia ###
 && LIBRARY_PATH=$(echo "$LIBRARY_PATH" | sed 's|^:||; s|:$||; s|::|:|g;') \
 && luarocks install http://raw.githubusercontent.com/jpuigcerver/Laia/master/rocks/laia-scm-1.rockspec
