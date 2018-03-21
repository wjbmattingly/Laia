#!/bin/sh

set -ex

CUDA="8.0";
CUDNN="cudnn5-devel";
OS="ubuntu16.04";

mkdir -p logs;

### CUDA Torch ###
{ DS=$(date +%s);
  nvidia-docker build -t mauvilsa/torch-cuda:$CUDA-$OS --build-arg CUDA_TAG=$CUDA-$CUDNN-$OS torch-cuda;
  echo "time: $(( $(date +%s) - DS )) seconds";
} 2>logs/torch-cuda$CUDA-$OS.err >logs/torch-cuda$CUDA-$OS.log;

### Laia ###
REV=$(git log --date=iso ../laia/Version.lua Dockerfile laia-docker | sed -n '/^Date:/{s|^Date: *||;s| .*||;s|-|.|g;p;}' | sort -r | head -n 1);
{ DS=$(date +%s);
  #nvidia-docker build --no-cache -t mauvilsa/laia:$REV-cuda$CUDA-$OS --build-arg TORCH_CUDA_TAG=$CUDA-$OS -f ./Dockerfile ..;
  nvidia-docker build -t mauvilsa/laia:$REV-cuda$CUDA-$OS --build-arg TORCH_CUDA_TAG=$CUDA-$OS -f ./Dockerfile ..;
  echo "time: $(( $(date +%s) - DS )) seconds";
} 2>logs/laia-cuda$CUDA-$OS.err >logs/laia-cuda$CUDA-$OS.log;
