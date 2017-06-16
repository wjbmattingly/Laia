#!/bin/sh

set -ex

#CUDA="8.0"; CUDA_SED="s|7\.5|8.0|g";
CUDA="7.5"; CUDA_SED="s|8\.0|7.5|g";
sed "$CUDA_SED" -i Dockerfile torch-cuda/Dockerfile;

#OS="ubuntu14.04";
OS="ubuntu14.04";
sed "s|ubuntu1.\.04|$OS|g" -i Dockerfile torch-cuda/Dockerfile;

mkdir -p logs;

### CUDA Torch ###
{ DS=$(date +%s);
  nvidia-docker build -t torch-cuda:$CUDA-$OS --build-arg CUDA=$CUDA torch-cuda/all
  echo "time: $(( $(date +%s) - DS )) seconds";
} 2>logs/torch-cuda$CUDA-$OS.err >logs/torch-cuda$CUDA-$OS.log;

### Laia ###
REV=$(sed -rn '/^Version.DATE/{ s|.*Date: *([0-9]+)-([0-9]+)-([0-9]+).*|\1.\2.\3|; p; }' ../laia/Version.lua);
{ DS=$(date +%s);
  nvidia-docker build -t laia:${REV}-cuda$CUDA-$OS --build-arg CUDA=$CUDA .;
  echo "time: $(( $(date +%s) - DS )) seconds";
} 2>logs/laia-cuda$CUDA-$OS.err >logs/laia-cuda$CUDA-$OS.log;
