#!/bin/bash
set -e;
export LC_NUMERIC=C;
export LUA_PATH="$(pwd)/../../?/init.lua;$(pwd)/../../?.lua;$LUA_PATH";

overwrite=false;
batch_size=16;
height=64;
experiments=(htr);

# Directory where the run.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)" != "$SDIR" ] && \
    echo "Please, run this script from the experiment top directory!" >&2 && \
    exit 1;

for c in ${experiments[@]}; do
  # Count the number of symbols to use in the output layer of the model.
  # Note: We subtract 1 from the list, because the symbols list includes the
  # <eps> symbol typically used by Kaldi.
  num_symbols=$[$(wc -l data/$c/lang/char/symbs.txt | cut -d\  -f1) - 1];

  # Create model & train it!
  [ -f model_lstm2d_$c.t7 -a "$overwrite" = false ] || {
    ../../laia-create-model2 \
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
      1 "$height" "$num_symbols" model_lstm2d_$c.t7;

    ../../laia-train-ctc \
      --use_distortions false \
      --batch_size "$batch_size" \
      --progress_table_output train_lstm2d_$c.dat \
      --early_stop_epochs 50 \
      --learning_rate 0.00027 \
      --log_also_to_stderr info \
      --log_level info \
      --log_file train_lstm2d_$c.log \
      --display_progress_bar true \
      model_lstm2d_$c.t7 data/$c/lang/char/symbs.txt \
      data/$c/tr.lst data/$c/lang/char/tr.txt  \
      data/$c/va.lst data/$c/lang/char/va.txt;
  }
done;
