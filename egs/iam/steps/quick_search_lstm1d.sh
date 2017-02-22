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
batch_size=6;
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

for f in "data/lists/$partition/tr_h$height.lst" \
	 "data/lang/char/$partition/tr.txt" \
	 "train/$ptype/syms.txt"; do
  [ ! -s "$f" ] && echo "ERROR: File \"$f\" does not exist!" >&2 && exit 1;
done;

# Get number of symbols
num_syms="$(tail -n1 train/$ptype/syms.txt | awk '{ print $2 }')";

# Create directory
tmpd="$(mktemp -d)";

shuf "data/lists/$partition/tr_h$height.lst" | head -n 32 | sort > "$tmpd/list";

echo "Check progress at $tmpd/dat ...";

multipliers=(4 8 12 16 20 24 32);
while true; do
  num_cnn_layers="$(shuf -i 3-5 -n 1)";
  num_cnn_hpool="$(shuf -i "2-$num_cnn_layers" -n 1)";
  num_cnn_vpool="$(shuf -i "2-$num_cnn_layers" -n 1)";
  cnn_multiplier="$(shuf -i 1-${#multipliers[@]} -n 1)";
  cnn_multiplier="${multipliers[cnn_multiplier - 1]}";

  cnn_num_features=();
  cnn_maxpool_size=();
  for i in $(seq 1 $num_cnn_layers); do
    cnn_num_features+=( $[i * cnn_multiplier] );
    if [ $i -le $num_cnn_hpool -a $i -le $num_cnn_vpool ]; then
      cnn_maxpool_size+=(2,2);
    elif [ $i -le $num_cnn_hpool ]; then
      cnn_maxpool_size+=(2,1);
    elif [ $i -le $num_cnn_vpool ]; then
      cnn_maxpool_size+=(1,2);
    else
      cnn_maxpool_size+=(0);
    fi;
  done;

  fm="$(printf "%25s %25s" "${cnn_num_features[*]}" "${cnn_maxpool_size[*]}")"
  [ -s "$tmpd/search" ] && grep -q "$fm" "$tmpd/search" && continue;

  # Create model
  ../../laia-create-model \
    --cnn_type leakyrelu \
    --cnn_kernel_size 3 \
    --cnn_num_features "${cnn_num_features[@]}" \
    --cnn_maxpool_size "${cnn_maxpool_size[@]}" \
    --cnn_batch_norm false \
    --rnn_num_layers 3 \
    --rnn_num_units 128 \
    --rnn_dropout 0 \
    --linear_dropout 0 \
    1 "$height" "$num_syms" "$tmpd/model";

  # Train model
  ../../laia-train-ctc \
    --use_distortions false \
    --batch_size "$batch_size" \
    --best_criterion train_cer \
    --early_stop_epochs 200 \
    --early_stop_threshold 0.05 \
    --learning_rate 0.0005 \
    --log_file "$tmpd/log" \
    --max_epochs 1000 \
    --progress_table_output "$tmpd/dat" \
    "$tmpd/model" "train/$ptype/syms.txt" \
    "$tmpd/list"   "data/lang/char/$partition/tr.txt";

  log=( $(awk '$NF == "*"{ print $3, $1, $2; }' "$tmpd/dat" | tail -n1) );
  printf "%25s %25s | %8.4f %5d %8.5f\n" \
	 "${cnn_num_features[*]}" "${cnn_maxpool_size[*]}" "${log[@]}" |
  tee -a "$tmpd/search";
done;

exit 0;
