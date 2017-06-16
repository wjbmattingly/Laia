#!/bin/sh

set -ex

CUDA="8.0";
sed 's|7\.5|8.0|g' -i $(find . -name Dockerfile);

#CUDA="7.5";
#sed 's|8\.0|7.5|g' -i $(find . -name Dockerfile);

OS="ubuntu16.04";
#OS="ubuntu14.04";
sed "s|ubuntu1.\.04|$OS|g" -i cuda-torch/deps1/Dockerfile;

mkdir -p logs;

### CUDA Torch dependencies 1 ###
{ DS=$(date +%s);
  nvidia-docker build -t cuda-torch-deps1:$CUDA --build-arg CUDA=$CUDA cuda-torch/deps1
  echo "time: $(( $(date +%s) - DS )) seconds";
} 2>logs/cuda-torch-deps1-$CUDA.err >logs/cuda-torch-deps1-$CUDA.log;

### CUDA Torch dependencies 2 ###
{ DS=$(date +%s);
  nvidia-docker build -t mauvilsa/cuda-torch-deps2:$CUDA --build-arg CUDA=$CUDA cuda-torch/deps2
  echo "time: $(( $(date +%s) - DS )) seconds";
} 2>logs/cuda-torch-deps2-$CUDA.err >logs/cuda-torch-deps2-$CUDA.log;

### CUDA Torch ###
{ DS=$(date +%s);
  nvidia-docker build -t mauvilsa/cuda-torch:$CUDA --build-arg CUDA=$CUDA cuda-torch
  echo "time: $(( $(date +%s) - DS )) seconds";
} 2>logs/cuda-torch-$CUDA.err >logs/cuda-torch-$CUDA.log;

### Laia ###
REV=$(sed -rn '/^Version.DATE/{ s|.*Date: *([0-9]+)-([0-9]+)-([0-9]+).*|\1.\2.\3|; p; }' ../laia/Version.lua);
{ DS=$(date +%s);
  nvidia-docker build -t mauvilsa/laia:${REV}-cuda$CUDA --build-arg CUDA=$CUDA .;
  echo "time: $(( $(date +%s) - DS )) seconds";
} 2>logs/laia-cuda$CUDA.err >logs/laia-cuda$CUDA.log;
