#!/bin/sh

set -ex

CUDA="8.0"; CUDA_SED="s|7\.5|8.0|g";
#CUDA="7.5"; CUDA_SED="s|8\.0|7.5|g";
sed "$CUDA_SED" -i Dockerfile torch-cuda/Dockerfile;

OS="ubuntu16.04";
#OS="ubuntu14.04";
sed "s|ubuntu1.\.04|$OS|g" -i Dockerfile torch-cuda/Dockerfile;

mkdir -p logs;

### CUDA Torch ###
{ DS=$(date +%s);
  nvidia-docker build -t mauvilsa/torch-cuda:$CUDA-$OS --build-arg CUDA=$CUDA torch-cuda;
  echo "time: $(( $(date +%s) - DS )) seconds";
} 2>logs/torch-cuda$CUDA-$OS.err >logs/torch-cuda$CUDA-$OS.log;

### Laia ###
REV=$(git log --date=iso ../laia/Version.lua Dockerfile laia-docker | sed -n '/^Date:/{s|^Date: *||;s| .*||;s|-|.|g;p;}' | sort -r | head -n 1);
{ DS=$(date +%s);
  nvidia-docker build --no-cache -t mauvilsa/laia:${REV}-cuda$CUDA-$OS --build-arg CUDA=$CUDA .;
  echo "time: $(( $(date +%s) - DS )) seconds";
} 2>logs/laia-cuda$CUDA-$OS.err >logs/laia-cuda$CUDA-$OS.log;
