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
partition="lines/aachen";
help_message="
Usage: ${0##*/} [options]

Options:
  --height            : (type = integer, default = $height)
                        Use images rescaled to this height.
  --batch_size        : (type = integer, default = $batch_size)
                        Batch size for training.
  --overwrite         : (type = boolean, default = $overwrite)
                        Overwrite previously created files.
";
source "$(pwd)/utils/parse_options.inc.sh" || exit 1;

# Get "lines" or "sentences" from the full partition string (e.g. lines/aachen)
ptype="${partition%%/*}";

for f in  "data/lists/$partition/te_h$height.lst" \
	  "data/lists/$partition/tr_h$height.lst" \
	  "data/lists/$partition/va_h$height.lst" \
	  "data/lang/char/$partition/tr.txt" \
	  "data/lang/char/$partition/va.txt" \
	  "train/$ptype/syms.txt"; do
  [ ! -s "$f" ] && echo "ERROR: File \"$f\" does not exist!" >&2 && exit 1;
done;

# Get number of symbols
num_syms="$(tail -n1 train/$ptype/syms.txt | awk '{ print $2 }')";

# Create directory
mkdir -p "train/$partition";

# Create model
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
  1 "$height" "$num_syms" "train/$partition/lstm1d.t7";

# Train model
../../laia-train-ctc \
  --use_distortions false \
  --batch_size "$batch_size" \
  --progress_table_output "train/$partition/lstm1d.dat" \
  --early_stop_epochs 25 \
  --early_stop_threshold 0.05 \
  --learning_rate 0.0005 \
  --log_also_to_stderr info \
  --log_level info \
  --log_file "train/$partition/lstm1d.log" \
  "train/$partition/lstm1d.t7" "train/$ptype/syms.txt" \
  "data/lists/$partition/tr_h${height}.lst" "data/lang/char/$partition/tr.txt" \
  "data/lists/$partition/va_h${height}.lst" "data/lang/char/$partition/va.txt";

exit 0;
