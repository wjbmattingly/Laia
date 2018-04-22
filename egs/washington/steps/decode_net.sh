#!/bin/bash
set -e;
export LC_NUMERIC=C;
export LUA_PATH="$(pwd)/../../?/init.lua;$(pwd)/../../?.lua;$LUA_PATH";

# Directory where the prepare.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/steps" != "$SDIR" ] && \
    echo "Please, run this script from the experiment top directory!" >&2 && \
  exit 1;

batch_size=16;
overwrite=false;
help_message="
Usage: ${0##*/} [options]

Options:
  --batch_size : (type = integer, default = $batch_size)
                 Batch size.
  --overwrite  : (type = boolean, default = $overwrite)
                 Overwrite previously created files.
";
source "$(pwd)/utils/parse_options.inc.sh" || exit 1;

## Get character-level transcript hypotheses
mkdir -p decode/no_lm/{char,word}/{cv1,cv2,cv3,cv4};
for cv in cv1 cv2 cv3 cv4; do
  for p in va te; do
    [[ "$overwrite" = false && -s decode/no_lm/char/$cv/$p.txt && \
      train/$cv/model.t7 -ot decode/no_lm/char/$cv/$p.txt ]] ||
    ../../laia-decode \
      --batch_size "$batch_size" \
      --log_level info \
      --symbols_table data/lang/char/syms.txt \
      train/$cv/model.t7 data/lists/$cv/$p.txt \
      > decode/no_lm/char/$cv/$p.txt ||
    exit 1;
  done;
done;

for cv in cv1 cv2 cv3 cv4; do
  mkdir -p data/lang/word/$cv;
  # Convert reference
  for p in te tr va; do
    [ "$overwrite" = false -a -s "data/lang/word/$cv/$p.txt" ] ||
    gawk '{
      printf("%s ", $1);
      for (i=2;i<=NF;++i) {
        if ($i == "<space>")   printf(" ");
        else if ($i == "s_bl") printf("(");
        else if ($i == "s_br") printf(")");
        else if ($i == "s_cm") printf(",")
        else if ($i == "s_et") printf("V");
        else if ($i == "s_lb") printf("L");
        else if ($i == "s_mi") printf("-");
        else if ($i == "s_pt") printf(".");
        else if ($i == "s_qo") printf(":");
        else if ($i == "s_qt") printf("'\''");
        else if ($i == "s_s")  printf("s");
        else if ($i == "s_sl") printf("sl");
        else if ($i == "s_sq") printf(";");
        else if (match($i, /^s_(.+)$/, A)) printf("%s", A[1]);
        else printf("%s", $i);
      }
      printf("\n");
    }' data/lang/char/$cv/$p.txt > data/lang/word/$cv/$p.txt;
  done;

  for p in te va; do
    [ "$overwrite" = false -a -s "decode/no_lm/word/$cv/$p.txt" ] ||
    gawk '{
      printf("%s ", $1);
      for (i=2;i<=NF;++i) {
        if ($i == "<space>")   printf(" ");
        else if ($i == "s_bl") printf("(");
        else if ($i == "s_br") printf(")");
        else if ($i == "s_cm") printf(",")
        else if ($i == "s_et") printf("V");
        else if ($i == "s_lb") printf("L");
        else if ($i == "s_mi") printf("-");
        else if ($i == "s_pt") printf(".");
        else if ($i == "s_qo") printf(":");
        else if ($i == "s_qt") printf("'\''");
        else if ($i == "s_s")  printf("s");
        else if ($i == "s_sl") printf("sl");
        else if ($i == "s_sq") printf(";");
        else if (match($i, /^s_(.+)$/, A)) printf("%s", A[1]);
        else printf("%s", $i);
      }
      printf("\n");
    }' decode/no_lm/char/$cv/$p.txt > decode/no_lm/word/$cv/$p.txt;
  done;
done;


## Compute CER.
if $(which compute-wer-bootci &> /dev/null); then
  for cv in cv1 cv2 cv3 cv4; do
    echo -n "$cv Valid ";
    compute-wer-bootci --mode=strict --print-args=false \
      ark:data/lang/char/$cv/va.txt ark:decode/no_lm/char/$cv/va.txt |
    grep WER | sed -r 's|Set1: ||g;s|95% Conf Interval ||g;s|%WER|%CER|g';

    echo -n "$cv Test ";
    compute-wer-bootci --mode=strict --print-args=false \
      ark:data/lang/char/$cv/te.txt ark:decode/no_lm/char/$cv/te.txt |
    grep WER | sed -r 's|Set1: ||g;s|95% Conf Interval ||g;s|%WER|%CER|g';
  done;
  echo "";
  for cv in cv1 cv2 cv3 cv4; do
    echo -n "$cv Valid ";
    compute-wer-bootci --mode=strict --print-args=false \
      ark:data/lang/word/$cv/va.txt ark:decode/no_lm/word/$cv/va.txt |
    grep WER | sed -r 's|Set1: ||g;s|95% Conf Interval ||g';

    echo -n "$cv Test ";
    compute-wer-bootci --mode=strict --print-args=false \
      ark:data/lang/word/$cv/te.txt ark:decode/no_lm/word/$cv/te.txt |
    grep WER | sed -r 's|Set1: ||g;s|95% Conf Interval ||g';
  done;
else
  echo "ERROR: Kaldi's compute-wer-bootci was not found in your PATH!" >&2;
fi;
