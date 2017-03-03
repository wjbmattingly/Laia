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

batch_size=16;
cnn_batch_norm=true;
cnn_dropout=0;
cnn_maxpool_size="2,2 2,2 2,2 0";
cnn_num_features="12 24 48 96";
cnn_type=leakyrelu;
continue_train=false;
gpu=1;
height=128;
linear_dropout=0.5;
overwrite=false;
partition="lines/aachen";
rnn_num_layers=4;
rnn_num_units=256;
rnn_dropout=0.5;
use_distortions=true;
model_name="lstm1d_h${height}";
help_message="
Usage: ${0##*/} [options]

Options:
  --batch_size  : (type = integer, default = $batch_size)
                  Batch size for training.
  --height      : (type = integer, default = $height)
                  Use images rescaled to this height.
  --overwrite   : (type = boolean, default = $overwrite)
                  Overwrite previously created files.
  --partition   : (type = string, default = \"$partition\")
                  Select the \"lines\" or \"sentences\" partition and the lists
                  to use (\"aachen\", \"original\" or \"kws\").
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
[ "$overwrite" = false -a "$continue_train" = true -a \
  -s "train/$partition/$model_name.t7" ] ||
../../laia-create-model \
  --cnn_batch_norm $cnn_batch_norm \
  --cnn_dropout $cnn_dropout \
  --cnn_kernel_size 3 \
  --cnn_maxpool_size $cnn_maxpool_size \
  --cnn_num_features $cnn_num_features \
  --cnn_type $cnn_type \
  --rnn_dropout "$rnn_dropout" \
  --rnn_num_layers "$rnn_num_layers" \
  --rnn_num_units "$rnn_num_units" \
  --linear_dropout "$linear_dropout" \
  --log_also_to_stderr info \
  --log_file "train/$partition/$model_name.log" \
  --log_level info \
  1 "$height" "$num_syms" "train/$partition/$model_name.t7";

# Train model
[ "$overwrite" = false -a "train/$partition/$model_name.t7" ] ||
../../laia-train-ctc \
  --batch_size "$batch_size" \
  --continue_train "$continue_train" \
  --use_distortions "$use_distortions" \
  --progress_table_output "train/$partition/$model_name.dat" \
  --early_stop_epochs 50 \
  --learning_rate 0.0003 \
  --learning_rate_decay 0.98 \
  --learning_rate_decay_after 100 \
  --learning_rate_decay_min 0.0001 \
  --log_also_to_stderr info \
  --log_level info \
  --log_file "train/$partition/$model_name.log" \
  --display_progress_bar true \
  --gpu "$gpu" \
  "train/$partition/$model_name.t7" "train/$ptype/syms.txt" \
  "data/lists/$partition/tr_h${height}.lst" "data/lang/char/$partition/tr.txt" \
  "data/lists/$partition/va_h${height}.lst" "data/lang/char/$partition/va.txt";

exit 0;
