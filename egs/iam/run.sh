#!/bin/bash
set -e;
export LC_NUMERIC=C;
export LUA_PATH="$(pwd)/../../?.lua;$LUA_PATH";

overwrite=false;
batch_size=1;

# Directory where the run.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)" != "$SDIR" ] && \
    echo "Please, run this script from the experiment top directory!" && \
    exit 1;

./scripts/prepare.sh --overwrite "$overwrite";


[ -f model.t7 -a "$overwrite" = false ] || \
    ../../create_model.lua \
    -cnn_type leakyrelu \
    -cnn_num_features "16 16 32 32" \
    -cnn_kernel_size  "3 3" \
    -cnn_maxpool_size "2 2 2 0" \
    -cnn_batch_norm false \
    -cnn_dropout 0 \
    -rnn_type blstm \
    -rnn_layers 3 \
    -rnn_units 256 \
    -rnn_dropout 0.5 \
    -linear_dropout 0.5 \
    -seed 74565 \
    1 64 79 model.t7;

[ -f train.log -a "$overwrite" = false ] || \
    ../../train.lua \
    -batch_size "$batch_size" \
    -early_stop_criterion valid_cer \
    -max_no_improv_epochs 15 \
    -min_relative_improv 0.0 \
    -adversarial_epsilon 0.007 \
    -adversarial_weight 0.0 \
    -grad_clip 5 \
    -weight_l1_decay 0 \
    -weight_l2_decay 0 \
    -alpha 0.95 \
    -learning_rate 0.005 \
    -learning_rate_decay 0.99 \
    -learning_rate_decay_after 10 \
    -gpu 0 \
    -seed 74565 \
    model.t7 data/lang/chars/symbs.txt \
    data/tr.mini.lst data/lang/chars/tr.txt \
    data/tr.mini.lst data/lang/chars/tr.txt;
#    data/tr.lst data/lang/chars/tr.txt \
#    data/va.lst data/lang/chars/va.txt;
