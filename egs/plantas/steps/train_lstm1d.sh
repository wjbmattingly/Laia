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

batch_chunk_size=0;
batch_size=16;
cnn_batch_norm=false;
cnn_dropout="0 0 0 0";
cnn_maxpool_size="2,2 2,2 2,2 0";
cnn_num_features="16 16 32 32";
cnn_type=leakyrelu;
continue_train=false;
early_stop_epochs=20;
gpu=1;
height=120;
learning_rate=0.0005;
linear_dropout=0.5;
model_name="lstm1d_h${height}";
overwrite=false;
rnn_num_layers=4;
rnn_num_units=256;
rnn_dropout=0.5;
use_distortions=true;
help_message="
Usage: ${0##*/} [options]

Options:
  --batch_chunk_size  : (type = integer, default = $batch_chunk_size)
  --batch_size        : (type = integer, default = $batch_size)
                        Batch size for training.
  --cnn_batch_norm    : (type = boolean, default = \"$cnn_batch_norm\")
  --cnn_dropout       : (type = float, default = \"$cnn_dropout\")
  --cnn_maxpool_size  : (type = integer, default = \"$cnn_maxpool_size\")
  --cnn_num_features  : (type = integer, default = \"$cnn_num_features\")
  --cnn_type          : (type = string, default = \"$cnn_type\")
  --continue_train    : (type = boolean, default = $continue_train)
  --early_stop_epochs : (type = integer, default = $early_stop_epochs)
  --gpu               : (type = integer, default = $gpu)
  --height            : (type = integer, default = $height)
                        Use images rescaled to this height.
  --learning_rate     : (type = float, default = $learning_rate)
  --linear_dropout    : (type = float, default = $linear_dropout)
  --model_name        : (type = string, default = \"$model_name\")
  --overwrite         : (type = boolean, default = $overwrite)
                        Overwrite previously created files.
  --rnn_num_layers    : (type = integer, default = $rnn_num_layers)
  --rnn_num_units     : (type = integer, default = $rnn_num_units)
  --rnn_dropout       : (type = float, default = $rnn_dropout)
  --use_distortions   : (type = boolean, default = \"$use_distortions\")
";
source "$(pwd)/utils/parse_options.inc.sh" || exit 1;
[ $# -ne 0 ] && echo "$help_message" >&2 && exit 1;

# Check required files
for f in "data/lists/tr_h$height.lst" \
	 "data/lists/va_h$height.lst" \
	 "data/lang/char/tr.txt" \
	 "data/lang/char/va_diplomatic.txt" \
	 "train/syms.txt"; do
  [ ! -s "$f" ] && echo "ERROR: File \"$f\" does not exist!" >&2 && exit 1;
done;

# Get number of symbols
num_syms="$(tail -n1 train/syms.txt | awk '{ print $2 }')";

if [[ "$overwrite" = true || ( ! -s "train/$model_name.t7" ) ]]; then
    # Create model
    [ "$continue_train" = true -a -s "train/$model_name.t7" ] ||
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
      --log_file "train/$model_name.log" \
      --log_level info \
      1 "$height" "$num_syms" "train/$model_name.t7";

    # Train model
    ../../laia-train-ctc \
      --batch_chunk_size "$batch_chunk_size" \
      --batch_size "$batch_size" \
      --normalize_loss false \
      --continue_train "$continue_train" \
      --use_distortions "$use_distortions" \
      --progress_table_output "train/$model_name.dat" \
      --early_stop_epochs "$early_stop_epochs" \
      --learning_rate "$learning_rate" \
      --log_also_to_stderr info \
      --log_level info \
      --log_file "train/$model_name.log" \
      --display_progress_bar true \
      --gpu "$gpu" \
      "train/$model_name.t7" "train/syms.txt" \
      "data/lists/tr_h${height}.lst" "data/lang/char/tr.txt" \
      "data/lists/va_h${height}.lst" "data/lang/char/va_diplomatic.txt";
fi;

exit 0;
