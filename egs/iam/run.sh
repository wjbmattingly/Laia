#!/bin/bash
set -e;
export LC_NUMERIC=C;
export LUA_PATH="$(pwd)/../../?/init.lua;$(pwd)/../../?.lua;$LUA_PATH";

overwrite=false;
batch_size=16;

# Directory where the run.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)" != "$SDIR" ] && \
    echo "Please, run this script from the experiment top directory!" && \
    exit 1;

./steps/prepare.sh --height 96 --overwrite "$overwrite";

[ -f model.t7 ] && cp model.t7 model.t7.bck;
[ -f train.log ] && cp train.log train.log.bck;

../../create_model.lua \
    -cnn_type leakyrelu \
    -cnn_num_features "16 16 32 32" \
    -cnn_kernel_size  "3 3 3 3" \
    -cnn_maxpool_size "2 2 2 0" \
    -cnn_batch_norm false \
    -cnn_dropout 0 \
    -rnn_type blstm \
    -rnn_layers 3 \
    -rnn_units 256 \
    -rnn_dropout 0.5 \
    -linear_dropout 0.5 \
    -seed 74565 \
    1 96 79 model.t7;

../../train.lua \
    -batch_size "$batch_size" \
    -early_stop_criterion valid_cer \
    -max_no_improv_epochs 50 \
    -min_relative_improv 0.0 \
    -adversarial_weight 0.0 \
    -grad_clip 5 \
    -weight_l1_decay 0 \
    -weight_l2_decay 0 \
    -alpha 0.95 \
    -learning_rate 0.001 \
    -learning_rate_decay 0.99 \
    -learning_rate_decay_after 10 \
    -gpu 0 \
    -seed 74565 \
    -output_progress train.log \
    model.t7 data/lang/chars/symbs.txt \
    data/tr.lst data/lang/chars/tr.txt \
    data/va.lst data/lang/chars/va.txt;

../../decode.lua \
    -batch_size "$batch_size" \
    -symbols_table data/lang/chars/symbs.txt \
    model.t7 data/te.lst | \
    compute-wer --text ark:data/lang/chars/te.txt ark:-;
