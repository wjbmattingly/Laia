#!/bin/bash
set -e;
export LC_NUMERIC=C;
export LUA_PATH="$(pwd)/../../?/init.lua;$(pwd)/../../?.lua;$LUA_PATH";

overwrite=false;
height=96;
num_labels=84;
batch_size=10;

# Directory where the run.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)" != "$SDIR" ] && \
    echo "Please, run this script from the experiment top directory!" && \
    exit 1;

mkdir -p data;

# download corpus
[ -f data/CScorpus_DB.tgz ] || \
    wget -P data/ https://www.prhlt.upv.es/projects/multimodal/cs/files/CScorpus_DB.tgz;
# extract it
[ -d data/corpus ] || \
    mkdir data/corpus && tar -xzf data/CScorpus_DB.tgz -C data/corpus;

./steps/prepare.sh --height "$height" --overwrite "$overwrite";

../../create_model.lua \
    -cnn_type leakyrelu \
    -cnn_num_features "32 64 96" \
    -cnn_kernel_size "3 3 5" \
    -cnn_maxpool_size "2,2 2,2 2,1" \
    -cnn_batch_norm false \
    -cnn_dropout 0 \
    -rnn_type blstm \
    -rnn_layers 2 \
    -rnn_units 128 \
    -rnn_dropout 0.5 \
    -linear_dropout 0.5 \
    -seed 74565 \
    1 "$height" "$num_labels" model.t7;

../../train.lua \
    -batch_size "$batch_size" \
    -early_stop_criterion valid_cer \
    -max_no_improv_epochs 50 \
    -adversarial_weight 0.0 \
    -grad_clip 5 \
    -weight_l1_decay 0 \
    -weight_l2_decay 0 \
    -alpha 0.95 \
    -learning_rate 0.001 \
    -learning_rate_decay 0.99 \
    -learning_rate_decay_after 50 \
    -gpu 0 \
    -seed 74565 \
    -output_progress train.log \
    model.t7 data/lang/chars/symbs.txt \
    data/train.lst data/lang/chars/train.txt \
    data/valid.lst data/lang/chars/train.txt;

../../decode.lua \
    -batch_size "$batch_size" \
    -symbols_table data/lang/chars/symbs.txt \
    model.t7 data/test.lst | \
    compute-wer --text ark:data/lang/chars/test.txt ark:-;
