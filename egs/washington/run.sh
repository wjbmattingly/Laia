#!/bin/bash
set -e;
export LC_NUMERIC=C;
export LUA_PATH="$(pwd)/../../?/init.lua;$(pwd)/../../?.lua;$LUA_PATH";

batch_size=16;
height=120;
overwrite=false;

## Directory where the run.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)" != "$SDIR" ] &&
echo "Please, run this script from the directory \"$SDIR\"!" >&2 &&
exit 1;

## Download lines images from FKI.
[ -f data/washingtondb-v1.0.zip ] || {
  [ -z "$FKI_USER" -o -z "$FKI_PASSWORD" ] &&
  echo "Please, set the FKI_USER and FKI_PASSWORD variables to download the" \
    "Washington database from the FKI servers." >&2 && exit 1;
  wget -P data --user="$FKI_USER" --password="$FKI_PASSWORD" \
    http://www.fki.inf.unibe.ch/DBs/iamHistDB/data/washingtondb-v1.0.zip;
}

## Unzip dataset.
[ -d data/washingtondb-v1.0 ] ||
( cd data && unzip -uqq washingtondb-v1.0.zip && rm -rf __MACOSX && cd - &> /dev/null ) ||
exit 1;

## Prepare data for Laia training.
./steps/prepare.sh --height "$height" --overwrite "$overwrite" || exit 1;

## This is the number of output symbols of the NN, including the <ctc> symbol.
num_symbols="$(tail -n1 data/lang/char/syms.txt | gawk '{print $2}')";

## Train on each partition.
mkdir -p train/{cv1,cv2,cv3,cv4};
for cv in cv1 cv2 cv3 cv4; do
  [[ "$overwrite" = false && -s train/$cv/model.t7 ]] && continue;
  th ../../laia-create-model \
    --cnn_batch_norm true \
    --cnn_num_features 16 32 64 64 \
    --cnn_maxpool_size 2 2 2 0 \
    --cnn_kernel_size 3 \
    --cnn_type leakyrelu \
    --rnn_num_units 128 \
    --rnn_num_layers 4 \
    --log_also_to_stderr info \
    --log_file train/$cv/log \
    --log_level info \
    -- 1 "$height" "$num_symbols" train/$cv/model.t7 &&
  th ../../laia-train-ctc \
    --batch_size "$batch_size" \
    --log_also_to_stderr info \
    --log_level info \
    --log_file train/$cv/log \
    --progress_table_output train/$cv/dat \
    --use_distortions true \
    --early_stop_epochs 100 \
    --learning_rate 0.0003 \
    train/$cv/model.t7 data/lang/char/syms.txt \
    data/lists/$cv/tr.txt data/lang/char/$cv/tr.txt \
    data/lists/$cv/va.txt data/lang/char/$cv/va.txt;
done;

./steps/decode_net.sh --overwrite "$overwrite"
