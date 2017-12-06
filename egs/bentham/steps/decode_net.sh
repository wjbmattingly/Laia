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
gpu=1;
height=120;
overwrite=false;
help_message="
Usage: ${0##*/} [options] model

Options:
  --batch_size  : (type = integer, default = $batch_size)
                  Batch size for training.
  --height      : (type = integer, default = $height)
                  Use images rescaled to this height.
  --overwrite   : (type = boolean, default = $overwrite)
                  Overwrite previously created files.
  --partition   : (type = string, default = \"$partition\")
                  Select \"aachen\" or \"kws\".
";
source "$(pwd)/utils/parse_options.inc.sh" || exit 1;
[ $# -ne 1 ] && echo "$help_message" >&2 && exit 1;
model="$1";
model_name="$(basename "$1" .t7)";

for f in  "data/lists/te_h$height.lst" \
	  "data/lists/va_h$height.lst" \
	  "data/lang/char/te.txt" \
	  "data/lang/char/va.txt" \
	  "data/lang/word/te.txt" \
	  "data/lang/word/va.txt" \
          "train/syms.txt" \
          "$model"; do
  [ ! -s "$f" ] && echo "ERROR: File \"$f\" does not exist!" >&2 && exit 1;
done;

hasR=0;
if which Rscript &> /dev/null; then hasR=1; fi;
hasComputeWer=0;
if which compute-wer &> /dev/null; then hasComputeWer=1; fi;

[ $hasR -ne 0 -o $hasComputeWer -ne 0 ] ||
echo "WARNING: Neither Rscript or compute-wer were found, so CER/WER won't be computed!" >&2;

mkdir -p decode/no_lm/{char,word};

tmpf="$(mktemp)";
for p in va te; do
  txt_char="decode/no_lm/char/${p}_${model_name}.txt";
  txt_word="decode/no_lm/word/${p}_${model_name}.txt";
  # Decode lines
  [ "$overwrite" = false -a -s "$txt_char" ] ||
  ../../laia-decode \
    --batch_size "$batch_size" \
    --symbols_table "train/syms.txt" \
    "$model" "data/lists/${p}_h$height.lst" > "$txt_char";
  # Get word-level transcript hypotheses for lines
  [ "$overwrite" = false -a -s "$txt_word" ] ||
  gawk '{
    printf("%s ", $1);
    for (i=2;i<=NF;++i) {
      if ($i == "@")
        printf(" ");
      else
        printf("%s", $i);
    }
    printf("\n");
  }' "$txt_char" > "$txt_word";
  if [ $hasR -eq 1 ]; then
    # Compute CER and WER with Confidence Intervals using R
    ./utils/compute-errors.py "data/lang/char/${p}.txt" "$txt_char" > "$tmpf";
    ./utils/compute-confidence.R "$tmpf" |
    gawk -v p="$p" '$1 == "%ERR"{
      printf("%CER %s: %.2f %s %s %s\n", p, $2, $3, $4, $5); }';
    ./utils/compute-errors.py "data/lang/word/${p}.txt" "$txt_word" > "$tmpf";
    ./utils/compute-confidence.R "$tmpf" |
    gawk -v p="$p" '$1 == "%ERR"{
      printf("%WER %s: %.2f %s %s %s\n", p, $2, $3, $4, $5); }';
  elif [ $hasComputeWer -eq 1 ]; then
    # Compute CER and WER using Kaldi's compute-wer
    compute-wer --text --mode=strict \
      "ark:data/lang/char/${p}.txt" "ark:$txt_char" 2>/dev/null |
    gawk -v p="$p" '$1 == "%WER"{ printf("%CER %s: %.2f\n", p, $2); }';
    compute-wer --text --mode=strict \
      "ark:data/lang/word/${p}.txt" "ark:$txt_word" 2>/dev/null |
    gawk -v p="$p" '$1 == "%WER"{ printf("%WER %s: %.2f\n", p, $2); }';
  fi;
done;
rm -f "$tmpf";
