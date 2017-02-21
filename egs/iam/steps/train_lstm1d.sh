#!/bin/bash
set -e;
export LC_NUMERIC=C;
export LUA_PATH="$(pwd)/../../?/init.lua;$(pwd)/../../?.lua;$LUA_PATH";

# Directory where the prepare.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/steps" != "$SDIR" ] && \
    echo "Please, run this script from the experiment top directory!" >&2 && \
    exit 1;
[ ! -f "$(pwd)/utils/parse_options.inc.sh" ] && \
    echo "Missing $(pwd)/utils/parse_options.inc.sh file!" >&2 && exit 1;

height=128;
batch_size=16;
overwrite=false;
help_message="
Usage: ${0##*/} [options] partition

Arguments:
  partition           : Id of the partition: \"aachen\" or \"original\".

Options:
  --height            : (type = integer, default = $height)
                        Use images rescaled to this height.
  --batch_size        : (type = integer, default = $batch_size)
                        Batch size for training.
  --overwrite         : (type = boolean, default = $overwrite)
                        Overwrite previously created files.
";
source "$(pwd)/utils/parse_options.inc.sh" || exit 1;
[ $# -ne 1 ] && echo "$help_message" >&2 && exit 1;

for f in "data/part/$1/te.lst" "data/part/$1/tr.lst" "data/part/$1/va.lst" \
  "data/lists/$1/te_h$height.lst" "data/lists/$1/tr_h$height.lst" \
  "data/lists/$1/va_h$height.lst" train/syms.txt; do
  [ ! -s "$f" ] && echo "ERROR: File \"$f\" does not exist!" >&2 && exit 1;
done;

num_syms="$(tail -n1 train/syms.txt | awk '{ print $2 }')";

mkdir -p "train/$1";


../../laia-create-model \
  --cnn_type leakyrelu \
  --cnn_kernel_size 3 \
  --cnn_num_features 12 24 48 96 \
  --cnn_maxpool_size 2,2 2,2 2,2 1,2 \
  --cnn_batch_norm false \
  --rnn_num_layers 4 \
  --rnn_num_units 256 \
  --rnn_dropout 0.5 \
  --linear_dropout 0.5 \
  --log_level info \
  1 "$height" "$num_syms" "train/$1/lstm1d.t7";

../../laia-train-ctc \
  --use_distortions false \
  --batch_size "$batch_size" \
  --progress_table_output "train/$1/lstm1d.dat" \
  --early_stop_epochs 250 \
  --early_stop_threshold 0.05 \
  --learning_rate 0.0005 \
  --log_also_to_stderr info \
  --log_level info \
  --log_file "train/$1/lstm1d.log" \
  --check_nan true \
  --check_inf true \
  --best_criterion train_cer \
  "train/$1/lstm1d.t7" train/syms.txt \
  mini1h.lst  "data/lang/char/$1/tr.txt";

:<<EOF
  "data/lists/$1/tr.lst" "data/lang/char/$1/tr.txt" \
  "data/lists/$1/va.lst" "data/lang/char/$1/va.txt";
EOF

exit 0;
