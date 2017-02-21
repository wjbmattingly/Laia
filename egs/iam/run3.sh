#!/bin/bash
set -e;
export LC_NUMERIC=C;
export LUA_PATH="$(pwd)/../../?/init.lua;$(pwd)/../../?.lua;$LUA_PATH";

overwrite=false;
batch_size=16;
height=64;

# Directory where the run.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)" != "$SDIR" ] &&
echo "Please, run this script from the experiment top directory!" >&2 &&
exit 1;

./steps/download.sh || exit 1;
./steps/prepare_images.sh --height "$height" --overwrite "$overwrite" || exit 1;
./steps/prepare_iam_text.sh --overwrite "$overwrite" || exit 1;
num_syms="$(tail -n1 train/syms.txt | awk '{ print $2 }')";

mkdir -p train/aachen;
../../laia-create-model \
  --cnn_type leakyrelu \
  --cnn_kernel_size 3 \
  --cnn_num_features 12 24 48 96 \
  --cnn_maxpool_size 2,2 2,2 1,2 1,2 \
  --cnn_batch_norm false \
  --rnn_num_layers 4 \
  --rnn_num_units 256 \
  --rnn_dropout 0.5 \
  --linear_dropout 0.5 \
  --log_level info \
  1 "$height" "$num_syms" train/aachen/h$height.t7;

../../laia-train-ctc \
  --use_distortions true \
  --batch_size "$batch_size" \
  --progress_table_output train/aachen/h$height.dat \
  --early_stop_epochs 25 \
  --early_stop_threshold 0.05 \
  --learning_rate 0.0005 \
  --log_also_to_stderr info \
  --log_level info \
  --log_file train/aachen/h$height.log \
  train/aachen/h$height.t7 train/syms.txt \
  data/lists/aachen/tr_h$height.lst data/lang/char/aachen/tr.txt \
  data/lists/aachen/va_h$height.lst data/lang/char/aachen/va.txt;

exit 0;
