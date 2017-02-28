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
# Directory where the prepare.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/steps" != "$SDIR" ] && \
    echo "Please, run this script from the experiment top directory!" >&2 && \
    exit 1;
[ ! -f "$(pwd)/utils/parse_options.inc.sh" ] && \
    echo "Missing $(pwd)/utils/parse_options.inc.sh file!" >&2 && exit 1;

batch_size=16;
gpu=1;
height=128;
overwrite=false;
partition="lines/aachen";
name="lstm1d_h${height}";
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
	  "data/lists/$partition/va_h$height.lst" \
	  "data/lang/char/$partition/te.txt" \
	  "data/lang/char/$partition/va.txt" \
	  "data/lang/word/$partition/te.txt" \
	  "data/lang/word/$partition/va.txt" \
          "train/$ptype/syms.txt" \
          "train/$partition/$name.t7"; do
  [ ! -s "$f" ] && echo "ERROR: File \"$f\" does not exist!" >&2 && exit 1;
done;

compute_wer="$(which compute-wer 2>/dev/null)" ||
echo "WARNING: compute-wer is not found in your PATH! CER/WER won't be computed!" >&2;

mkdir -p decode/no_lm/{char,word}/"$partition";

[ "$overwrite" = false -a -s "train/$partition/$name.prior" ] ||
../../laia-force-align \
  --batch_size "$batch_size" \
  "train/$partition/$name.t7" \
  "train/$ptype/syms.txt" \
  "data/lists/$partition/tr_h$height.lst" \
  "data/lang/char/$partition/tr.txt" \
  /dev/null "train/$partition/$name.prior";

for p in va te; do
  # Decode lines
  [ "$overwrite" = false -a \
    -s "decode/no_lm/char/$partition/${p}_${name}.txt" ] ||
  ../../laia-decode \
    --batch_size "$batch_size" \
    --symbols_table "train/$ptype/syms.txt" \
    "train/$partition/$name.t7" \
    "data/lists/$partition/${p}_h$height.lst" \
    > "decode/no_lm/char/$partition/${p}_${name}.txt";
  # Get word-level transcript hypotheses
  [ "$overwrite" = false -a \
    -s "decode/no_lm/word/$partition/${p}_${name}.txt" ] ||
  awk '{
    printf("%s ", $1);
    for (i=2;i<=NF;++i) {
      if ($i == "<space>")
        printf(" ");
      else
        printf("%s", $i);
    }
    printf("\n");
  }' "decode/no_lm/char/$partition/${p}_${name}.txt" \
    > "decode/no_lm/word/$partition/${p}_${name}.txt";
  # Compute CER and WER using Kaldi's compute-wer
  if [ -n "$compute_wer" ]; then
    "$compute_wer" --text --mode=strict \
      "ark:data/lang/char/$partition/${p}.txt" \
      "ark:decode/no_lm/char/$partition/${p}_${name}.txt" 2>/dev/null |
    awk -v p="$p" '$1 == "%WER"{ printf("%CER %s: %.2f\n", p, $2); }';
    "$compute_wer" --text --mode=strict \
      "ark:data/lang/word/$partition/${p}.txt" \
      "ark:decode/no_lm/word/$partition/${p}_${name}.txt" 2>/dev/null |
    awk -v p="$p" '$1 == "%WER"{ printf("%WER %s: %.2f\n", p, $2); }';
  fi;
done;
