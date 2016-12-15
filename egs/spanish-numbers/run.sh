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

[ -f model.t7 -a "$overwrite" = false ] || {
  th ../../laia-create-model \
    --cnn_batch_norm true \
    --cnn_type leakyrelu \
    -- 1 64 $[$(wc -l data/lang/char/symbs.txt) - 1] model.t7;

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
    model.t7 data/lang/char/symbs.txt \
    data/train.lst data/lang/char/train.txt \
    data/test.lst data/lang/char/test.txt;
}

# Decode
th ../../laia-decode --symbols_table data/lang/char/symbs.txt \
  model.t7 data/test.lst > test_hyp.char.txt;

# Check compute-wer
which compute-wer &> /dev/null || {
  echo "Kaldi's compute-wer was not found in your PATH!" >&2;
  exit 1;
}

# ... and compute CER
compute-wer --mode=strict ark:data/lang/char/test.txt ark:test_hyp.char.txt |
grep WER | sed -r 's|%WER|%CER|g';

# Get word-level transcript hypotheses
awk '{
  printf("%s ", $1);
  for (i=2;i<=NF;++i) {
    if ($i == "{space}")
      printf(" ");
    else
      printf("%s", $i);
  }
  printf("\n");
}' test_hyp.char.txt > test_hyp.word.txt;
# ... and compute WER
compute-wer --mode=strict ark:data/lang/word/test.txt ark:test_hyp.word.txt |
grep WER;


th ../../laia-netout --output_format lattice --output_transform logsoftmax \
  model.t7 data/test.lst test.lat;
