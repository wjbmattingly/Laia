#!/bin/bash
set -e;
export LC_NUMERIC=C;
export LUA_PATH="$(pwd)/../../?/init.lua;$(pwd)/../../?.lua;$LUA_PATH";

batch_size=16;
height=96;
overwrite=false;

# Directory where the run.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)" != "$SDIR" ] && \
  echo "Please, run this script from the experiment top directory!" && \
  exit 1;

mkdir -p data;

# download corpus
[ -f data/CScorpus_DB.tgz ] || \
  wget -P data/ https://www.prhlt.upv.es/projects/multimodal/cs/files/CScorpus_DB.tgz;
# extract it
[ -d data/corpus ] || \
  mkdir data/corpus && tar -xzf data/CScorpus_DB.tgz -C data/corpus;

./steps/prepare.sh --height "$height" --overwrite "$overwrite";
num_labels="$[$(wc -l data/lang/char/symbs.txt | cut -d\  -f1) - 1]";

[ -f model.t7 -a "$overwrite" = false ] || {
  # Create model
  ../../laia-create-model \
    --cnn_type leakyrelu \
    --cnn_num_features 32 64 96 \
    --cnn_kernel_size 3 3 5 \
    --cnn_maxpool_size 2,2 2,2 2,1 \
    --cnn_batch_norm false \
    --rnn_num_layers 2 \
    --rnn_num_units 128 \
    --rnn_dropout 0.5 \
    --linear_dropout 0.5 \
    1 "$height" "$num_labels" model.t7;

  # Train model
  ../../laia-train-ctc \
    --batch_size "$batch_size" \
    --log_also_to_stderr info \
    --log_level info \
    --log_file train.log \
    --progress_table_output train.dat \
    --use_distortions true \
    --learning_rate 0.0005 \
    --early_stop_epochs 50 \
    model.t7 data/lang/char/symbs.txt \
    data/train.lst data/lang/char/train.txt \
    data/valid.lst data/lang/char/train.txt;
}

mkdir -p decode/{char,word};

# Get char-level transcript hypotheses
../../laia-decode \
  --batch_size "$batch_size" \
  --symbols_table data/lang/char/symbs.txt \
  model.t7 data/test.lst > decode/char/test.txt;

# Get word-level transcript hypotheses
awk '{
  printf("%s ", $1);
  for (i=2;i<=NF;++i) {
    if ($i == "@")
      printf(" ");
    else
      printf("%s", $i);
  }
  printf("\n");
}' decode/char/test.txt > decode/word/test.txt;

# Compute CER/WER.
if $(which compute-wer &> /dev/null); then
  compute-wer --mode=strict \
    ark:data/lang/char/test.txt ark:decode/char/test.txt |
  grep WER | sed -r 's|%WER|%CER|g';

  compute-wer --mode=strict \
    ark:data/lang/word/test.orig.txt ark:decode/word/test.txt |
  grep WER;
else
  echo "ERROR: Kaldi's compute-wer was not found in your PATH!" >&2;
fi;
