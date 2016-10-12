#!/bin/bash
set -e;
export LC_NUMERIC=C;
export LUA_PATH="$(pwd)/../../?/init.lua;$(pwd)/../../?.lua;$LUA_PATH";

overwrite=true;
batch_size=16;

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

./steps/prepare.sh --height 96 --overwrite "$overwrite";

exit 0;
../../create_model.lua \
    -cnn_type leakyrelu \
    -cnn_maxpool_size "2,2 2,2 2,2 0" \
    -cnn_batch_norm false \
    -cnn_dropout 0 \
    -rnn_type blstm \
    -rnn_layers 3 \
    -rnn_units 256 \
    -rnn_dropout 0.5 \
    -linear_dropout 0.5 \
    -seed 74565 \
    1 64 84 model.t7;

../../train.lua \
    -batch_size "$batch_size" \
    -early_stop_criterion valid_cer \
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
    data/train.lst data/lang/chars/train.txt \
    data/valid.lst data/lang/chars/train.txt;
