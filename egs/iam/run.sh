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

./steps/prepare.sh --height "$height" --overwrite "$overwrite";

function rand_choice () {
  local i="$(shuf -i 1-$# -n1)";
  local a=( "$@" );
  echo ${a[i-1]};
}

for c in ${experiments[@]}; do
  # Keep a copy of the previous trained model, just in case you mess something.
  [ -f train_$c.log ] && cp train_$c.log train_$c.log.bck;
  [ -f train_$c.dat ] && cp train_$c.dat train_$c.dat.bck;
  [ -f model_$c.t7 ] && cp model_$c.t7 model_$c.t7.bck;

  # Count the number of symbols to use in the output layer of the model.
  # Note: We subtract 1 from the list, because the symbols list includes the
  # <eps> symbol typically used by Kaldi.
  num_symbols=$[$(wc -l data/$c/lang/char/symbs.txt | cut -d\  -f1) - 1];

  # Create a new model.
  ../../laia-create-model \
    --cnn_type leakyrelu \
    --cnn_num_features 16 16 32 64 \
    --cnn_kernel_size 3 \
    --cnn_maxpool_size 2,2 2,2 2,2 2,1 \
    --cnn_batch_norm true \
    --cnn_dropout 0 \
    --rnn_type blstm \
    --rnn_num_layers 3 \
    --rnn_num_units 256 \
    --rnn_dropout 0.5 \
    --linear_dropout 0.5 \
    -- 1 "$height" "$num_symbols" model_$c.t7;

  ../../laia-train-ctc \
    --use_distortions true \
    --batch_size "$batch_size" \
    --progress_table_output train_$c.dat \
    --log_also_to_stderr info \
    --log_level info \
    --log_file train_$c.log \
    --early_stop_epochs 50 \
    --learning_rate 0.001 \
    model_$c.t7 data/$c/lang/char/symbs.txt \
    data/$c/tr.lst data/$c/lang/char/tr.txt \
    data/$c/va.lst data/$c/lang/char/va.txt;

  : <<EOF
  # Create a new model.
  for r in $(seq 1 1000); do
    num_layers=$(shuf -i 3-5 -n1);
    cnn_maxpool_size=( "2,2" );
    cnn_num_features=( 32 );
    for i in $(seq 2 $num_layers); do
      cnn_maxpool_size+=( "$(rand_choice 0 2,2 2,1 1,2)" );
      cnn_num_features+=( 32 );
    done;

    ../../laia-create-model \
      --cnn_type leakyrelu \
      --cnn_num_features ${cnn_num_features[@]} \
      --cnn_kernel_size 3 \
      --cnn_maxpool_size ${cnn_maxpool_size[@]} \
      --cnn_batch_norm false \
      --cnn_dropout 0 \
      --rnn_type blstm \
      --rnn_num_layers 2 \
      --rnn_num_units 64 \
      --rnn_dropout 0 \
      --linear_dropout 0 \
      -- 1 "$height" "$num_symbols" model_$c.t7;

    ../../laia-train-ctc \
      --batch_size "$batch_size" \
      --progress_table_output train.$r.dat \
      --early_stop_epochs 400 \
      --max_epochs 1000 \
      --learning_rate 0.001 \
      model_$c.t7 data/$c/lang/char/symbs.txt \
      data/tr.mini.lst data/htr/lang/char/tr.txt \
      data/tr.mini.lst data/htr/lang/char/tr.txt;

    printf "%4d %30s %30s %s\n" "$r" \
      "${cnn_num_features[*]}" "${cnn_maxpool_size[*]}" \
      "$(awk '$NF == "*"{ print $5, $1 }' train.$r.dat | tail -n1)";
  done > model_selection.log;
EOF

done






: <<EOF
  ../../laia-train-ctc \
    --batch_size "$batch_size" \
    -early_stop_criterion valid_cer \
    -max_no_improv_epochs 50 \
    -min_relative_improv 0.0 \
    -adversarial_weight 0.0 \
    -grad_clip 5 \
    -weight_l1_decay 0 \
    -weight_l2_decay 0 \
    -alpha 0.95 \
    -learning_rate 0.001 \
    -learning_rate_decay 0.99 \
    -learning_rate_decay_after 10 \
    -gpu 0 \
    -seed 74565 \
    -output_progress train.log \
    model.t7 data/lang/chars/symbs.txt \
    data/tr.lst data/lang/chars/tr.txt \
    data/va.lst data/lang/chars/va.txt;

  ../../laia-decode.lua \
    -batch_size "$batch_size" \
    -symbols_table data/lang/chars/symbs.txt \
    model.t7 data/te.lst | \
    compute-wer --text ark:data/lang/chars/te.txt ark:-;
done;

EOF
