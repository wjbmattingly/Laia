#!/bin/bash
set -e;
export LC_NUMERIC=C;
export LUA_PATH="$(pwd)/../../?/init.lua;$(pwd)/../../?.lua;$LUA_PATH";
export PATH="$(pwd)/../../:$PATH";

overwrite=false;
batch_size=16;

# Directory where the run.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)" != "$SDIR" ] && \
  echo "Please, run this script from the experiment top directory!" && \
  exit 1;

mkdir -p data/;
# download dataset
[ -f data/Spanish_Number_DB.tgz ] || \
  wget -P data/ https://www.prhlt.upv.es/corpora/spanish-numbers/Spanish_Number_DB.tgz;
# extract it
[ -d data/Spanish_Number_DB ] || \
  tar -xzf data/Spanish_Number_DB.tgz -C data/;

./steps/prepare.sh --overwrite "$overwrite";

th ../../laia-create-model \
  --cnn_batch_norm true \
  --cnn_type leakyrelu \
  -- 1 64 20 model.t7;

th ../../laia-train-ctc \
  --adversarial_weight 0.5 \
  --batch_size "$batch_size" \
  --log_also_to_stderr info \
  --log_level info \
  --log_file laia.log \
  --progress_table_output laia.dat \
  --use_distortions true \
  --early_stop_epochs 100 \
  --learning_rate 0.0005 \
  model.t7 data/lang/chars/symbs.txt \
  data/train.lst data/lang/chars/train.txt \
  data/test.lst data/lang/chars/test.txt;
