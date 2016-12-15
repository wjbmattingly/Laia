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

mkdir -p data;

## Download lines images from FKI.
[ -f data/lines.tgz -o -f data/imgs/a02-000-00.png ] || {
  [ -z "$FKI_USER" -o -z "$FKI_PASSWORD" ] && \
    echo "Please, set the FKI_USER and FKI_PASSWORD variables to download the" \
    "IAM database from the FKI servers." >&2 && exit 1;
  wget -P data --user="$FKI_USER" --password="$FKI_PASSWORD" \
    http://www.fki.inf.unibe.ch/DBs/iamDB/data/lines/lines.tgz;
}

## Put all images into a single directory.
mkdir -p data/imgs;
[ -f data/imgs/a02-000-00.png ] || {
  tar zxf data/lines.tgz -C data/imgs &&
  find data/imgs -name "*.png" | xargs -I{} mv {} data/imgs &&
  find data/imgs -name "???" | xargs rm -r;
}

## Prepare images
./steps/prepare.sh --height "$height" --overwrite "$overwrite";

for c in ${experiments[@]}; do
  # Keep a copy of the previous trained model, just in case you mess something.
  [ -f train_$c.log ] && cp train_$c.log train_$c.log.bck;
  [ -f train_$c.dat ] && cp train_$c.dat train_$c.dat.bck;
  [ -f model_$c.t7 ] && cp model_$c.t7 model_$c.t7.bck;

  # Count the number of symbols to use in the output layer of the model.
  # Note: We subtract 1 from the list, because the symbols list includes the
  # <eps> symbol typically used by Kaldi.
  num_symbols=$[$(wc -l data/$c/lang/char/symbs.txt | cut -d\  -f1) - 1];

  # Create model & train it!
  [ -f model_$c.t7 -a "$overwrite" = false ] || {
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
      1 "$height" "$num_symbols" model_$c.t7;

    ../../laia-train-ctc \
      --use_distortions true \
      --batch_size "$batch_size" \
      --progress_table_output train_$c.dat \
      --early_stop_epochs 50 \
      --learning_rate 0.00027 \
      --log_also_to_stderr info \
      --log_level info \
      --log_file train_$c.log \
      model_$c.t7 data/$c/lang/char/symbs.txt \
      data/$c/tr.lst data/$c/lang/char/tr.txt  \
      data/$c/va.lst data/$c/lang/char/va.txt;
  }

  mkdir -p decode/$c/{char,word};

  # Get character-level transcript hypotheses
  ../../laia-decode \
    --batch_size "$batch_size" \
    --log_level info \
    --symbols_table data/$c/lang/char/symbs.txt \
    model_$c.t7 data/$c/te.lst > decode/$c/char/te.txt;

  # Get word-level transcript hypotheses
  awk '{
    printf("%s ", $1);
    for (i=2;i<=NF;++i) {
      if ($i == "<space>")
        printf(" ");
      else
        printf("%s", $i);
    }
    printf("\n");
  }' decode/$c/char/te.txt > decode/$c/word/te.txt;

  # Compute CER/WER.
  if $(which compute-wer &> /dev/null); then
    compute-wer --mode=strict \
      ark:data/$c/lang/char/te.txt ark:decode/$c/char/te.txt |
    grep WER | sed -r 's|%WER|%CER|g';

    compute-wer --mode=strict \
      ark:data/$c/lang/word/te.txt ark:decode/$c/word/te.txt |
    grep WER;
  else
    echo "ERROR: Kaldi's compute-wer was not found in your PATH!" >&2;
  fi;
done;
