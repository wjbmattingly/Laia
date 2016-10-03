#!/bin/bash
set -e;
export LC_NUMERIC=C;
export LUA_PATH="$(pwd)/../../?/init.lua;$(pwd)/../../?.lua;$LUA_PATH";
export PATH="$(pwd)/../../:$PATH";

overwrite=false;
batch_size=30;

# Directory where the run.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)" != "$SDIR" ] && \
  echo "Please, run this script from the experiment top directory!" && \
  exit 1;

mkdir -p data/;
# download dataset
wget -P data/ https://www.prhlt.upv.es/corpora/spanish-numbers/Spanish_Number_DB.tgz;
# extract it
tar -xzf data/Spanish_Number_DB.tgz -C data/;

./scripts/prepare.sh --overwrite "$overwrite";

create_model.lua \
  -cnn_type leakyrelu \
  1 64 20 model.t7;

train.lua \
  -batch_size "$batch_size" \
  -early_stop_criterion valid_cer \
  -max_no_improv_epochs 15 \
  -num_samples_epoch 4000 \
  -adversarial_weight 0.5 \
  -output_progress laia.log \
  model.t7 data/lang/chars/symbs.txt \
  data/train.lst data/lang/chars/train.txt \
  data/test.lst data/lang/chars/test.txt;
