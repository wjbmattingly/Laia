#!/bin/bash
set -e;
export LC_NUMERIC=C;
export LUA_PATH="$(pwd)/../../?/init.lua;$(pwd)/../../?.lua;$LUA_PATH";

# Directory where the run.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)" != "$SDIR" ] && \
    echo "Please, run this script from the experiment top directory!" >&2 && \
    exit 1;

# Step 1. Download data.
./steps/download.sh;

# Step 2. Prepare data.
./steps/prepare.sh;

# Step 3. Train network.
./steps/train_lstm1d.sh --model_name "lstm1d_h128";

# Step 4. Decode using only the neural network.
./steps/decode_net.sh "train/lstm1d_h128.t7";

# Step 5. Decode using a word n-gram LM.
./steps/decode_lm.sh "train/lstm1d_h128.t7";

exit 0;
