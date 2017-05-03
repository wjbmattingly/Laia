#!/bin/bash
set -e;
export LC_NUMERIC=C;
export LUA_PATH="$(pwd)/../../?/init.lua;$(pwd)/../../?.lua;$LUA_PATH";

batch_size=16;
overwrite=false;

# Directory where the run.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)" != "$SDIR" ] && \
    echo "Please, run this script from the experiment top directory!" >&2 && \
    exit 1;

# 1. Download data.
./steps/download.sh --overwrite "$overwrite";
# 2. Prepare images and transcripts for training.
./steps/prepare.sh --overwrite "$overwrite";
# 3. Train a neural network model.
./steps/train_lstm1d.sh --batch_size "$batch_size" --overwrite "$overwrite";
# 4. Decode using the trained model both validation and test data.
./steps/decode_net.sh --batch_size "$batch_size" --overwrite "$overwrite" \
  train/model.t7;

exit 0;
