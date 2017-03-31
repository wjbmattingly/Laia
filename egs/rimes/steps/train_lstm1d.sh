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
cnn_batch_norm=true;
cnn_dropout=0;
cnn_maxpool_size="2,2 2,2 2,2 0 0";
cnn_num_features="16 32 48 64 80";
cnn_type=leakyrelu;
continue_train=false;
early_stop_epochs=20;
gpu=1;
height=128;
learning_rate=0.0003;
linear_dropout=0.5;
overwrite=false;
rnn_num_layers=5;
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
";
source "$(pwd)/utils/parse_options.inc.sh" || exit 1;
[ $# -ne 0 ] && echo "$help_message" >&2 && exit 1;

# Check required files
for f in  "data/lists/te_h$height.lst" \
	  "data/lists/tr_h$height.lst" \
	  "data/lists/va_h$height.lst" \
	  "data/lang/lines/char/tr.txt" \
	  "data/lang/lines/char/va.txt" \
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
      "data/lists/tr_h${height}.lst" "data/lang/lines/char/tr.txt" \
      "data/lists/va_h${height}.lst" "data/lang/lines/char/va.txt";
fi;

exit 0;
