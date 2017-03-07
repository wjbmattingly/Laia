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
word_order=3;
char_order=10;
prior_scale=1.0;
voc_size=50000;
height=128;
partition="aachen";
overwrite=false;
help_message="
Usage: ${0##*/} [options] model

Options:
  --height      : (type = integer, default = $height)
                  Use images rescaled to this height.
  --overwrite   : (type = boolean, default = $overwrite)
                  Overwrite previously created files.
";
source utils/parse_options.inc.sh || exit 1;
[ $# -ne 1 ] && echo "$help_message" >&2 && exit 1;
model="$1";
model_name="$(basename "$1" .t7)";

# Check required files
for f in train/lines/syms.txt \
         "$model" \
         "data/lists/lines/$partition/tr_h${height}.lst" \
         "data/lists/lines/$partition/te_h${height}.lst" \
         "data/lists/lines/$partition/va_h${height}.lst" \
         "data/lang/char/lines/$partition/tr.txt" \
         "data/lang/char/lines/$partition/te.txt" \
         "data/lang/char/lines/$partition/va.txt" \
         "data/lang/word/lines/$partition/te.txt" \
         "data/lang/word/lines/$partition/va.txt"; do
  [ ! -s "$f" ] && echo "ERROR: File \"$f\" was not found!" >&2 && exit 1;
done;

mkdir -p "decode/lm/$partition/word_lm" "decode/lkh/$partition"/{lines,forms};

[ "$overwrite" = false -a -s "${model/.t7/.prior}" ] ||
../../laia-force-align \
  --batch_size "$batch_size" \
  "$model" \
  "train/lines/syms.txt" \
  "data/lists/lines/$partition/tr_h$height.lst" \
  "data/lang/char/lines/$partition/tr.txt" \
  /dev/null "${model/.t7/.prior}";

# Compute log-likelihoods from the network.
echo $prior_scale
for p in va te; do
  lines_ark="decode/lkh/$partition/lines/${p}_${model_name}_ps${prior_scale}.ark";
  forms_ark="decode/lkh/$partition/forms/${p}_${model_name}_ps${prior_scale}.ark";
  # LINE log-likelihoods
  [ "$overwrite" = false -a -s "$lines_ark" ] ||
  ../../laia-netout \
    --batch_size "$batch_size" \
    --prior "${model/.t7/.prior}" \
    --prior_alpha "$prior_scale" \
    "$model" "data/lists/lines/$partition/${p}_h${height}.lst" /dev/stdout |
  copy-matrix ark:- "ark:$lines_ark";
  # FORM log-likelihoods
  [ "$overwrite" = false -a -s "$forms_ark" ] ||
  ./utils/join_lines_arks.sh "train/lines/syms.txt" "$lines_ark" "$forms_ark";
done;


# Build lexicon from the boundaries files.
lexiconp=data/lang/word/lexiconp.txt;
[ "$overwrite" = false -a -s "$lexiconp" ] ||
./utils/prepare_word_lexicon_from_boundaries.sh \
  data/lang/word/forms/aachen/tr_boundaries.txt \
  data/lang/word/external/*_boundaries.txt > "$lexiconp" ||
{ echo "ERROR: Creating file \"$lexiconp\"!" >&2 && exit 1; }

# Build word-level language model WITHOUT the unknown token
./utils/build_word_lm.sh --order "$word_order" --voc_size "$voc_size" \
  --unknown false --srilm_options "-kndiscount -interpolate" \
  --overwrite "$overwrite" \
  data/lang/word/forms/aachen/tr_{tokenized,boundaries}.txt \
  data/lang/word/forms/aachen/va_{tokenized,boundaries}.txt \
  data/lang/word/forms/aachen/te_{tokenized,boundaries}.txt \
  data/lang/word/external/lob_excludealltestsets_{tokenized,boundaries}.txt \
  data/lang/word/external/brown_{tokenized,boundaries}.txt \
  data/lang/word/external/wellington_{tokenized,boundaries}.txt \
  "decode/lm/$partition/word_lm";

./utils/build_word_fsts.sh \
  train/lines/syms.txt data/lang/word/lexiconp.txt \
  "decode/lm/$partition/word_lm/interpolation-${word_order}gram-${voc_size}.arpa.gz" \
  "decode/lm/$partition/word_fst-${word_order}gram-${voc_size}";

exit 0;

# Build word-level language model with the unknown token
./utils/build_word_lm.sh --order "$word_order" --voc_size "$voc_size" \
  --unknown true --srilm_options "-kndiscount -interpolate" \
  --overwrite "$overwrite" \
  data/lang/word/forms/aachen/tr_{tokenized,boundaries}.txt \
  data/lang/word/forms/aachen/va_{tokenized,boundaries}.txt \
  data/lang/word/forms/aachen/te_{tokenized,boundaries}.txt \
  data/lang/word/external/lob_excludealltestsets_{tokenized,boundaries}.txt \
  data/lang/word/external/brown_{tokenized,boundaries}.txt \
  data/lang/word/external/wellington_{tokenized,boundaries}.txt \
  decode/lm/word_lm_unk;

# Build char-level language model for backoff
./utils/build_char_backoff_lm.sh \
  --exclude_vocab "decode/lm/word_lm_unk/voc-$voc_size" --order "$char_order" \
  --srilm_options "-wbdiscount -interpolate" --overwrite "$overwrite" \
  train/lines/syms.txt \
  data/lang/word/forms/aachen/{tr,va,te}_tokenized.txt \
  data/lang/word/external/*_tokenized.txt \
  decode/lm/char_backoff_lm;



#./utils/build_word_fsts.sh
