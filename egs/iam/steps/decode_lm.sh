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

acoustic_scales="1.39 1.18";
batch_size=16;
beam=50;
char_order=10;
height=128;
max_active=2147483647;
overwrite=false;
overwrite_align=false;
overwrite_lexicon=false;
overwrite_likelihoods=false;
overwrite_char_decode=false;
overwrite_char_fst=false;
overwrite_char_lm=false;
overwrite_word_decode=false;
overwrite_word_fst=false;
overwrite_word_lm=false;
partition="aachen";
prior_scale=0.2;
qsub_opts="";
voc_size=50000;
word_order=3;
help_message="
Usage: ${0##*/} [options] model

Options:
  --height        : (type = integer, default = $height)
                    Use images rescaled to this height.
  --overwrite_all : (type = boolean, default = $overwrite)
                    Overwrite ALL previous stages.
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
         "data/lang/char/forms/$partition/te.txt" \
         "data/lang/char/forms/$partition/va.txt" \
         "data/lang/word/forms/$partition/te.txt" \
         "data/lang/word/forms/$partition/va.txt"; do
  [ ! -s "$f" ] && echo "ERROR: File \"$f\" was not found!" >&2 && exit 1;
done;

hasR=0;
if which Rscript &> /dev/null; then hasR=1; fi;
hasComputeWer=0;
if which compute-wer &> /dev/null; then hasComputeWer=1; fi;

[ $hasR -ne 0 -o $hasComputeWer -ne 0 ] ||
echo "WARNING: Neither Rscript or compute-wer were found, so CER/WER won't be computed!" >&2;

mkdir -p "decode/lm/$partition" "decode/lkh/$partition"/{lines,forms};

if [ "$overwrite" = true ]; then
  overwrite_align=true;
  overwrite_likelihoods=true;
  overwrite_lexicon=true;
  overwrite_word_lm=true;
  overwrite_word_fst=true;
  overwrite_word_decode=true;
  overwrite_char_lm=true;
  overwrite_char_fst=true;
  overwrite_char_decode=true;
fi;

# Do forced alignment.
[ "$overwrite_align" = false -a -s "${model/.t7/.prior}" ] || {
  echo "Forced alignment can take a while..." >&2 &&
  overwrite_likelihoods=true;
  ../../laia-force-align --batch_size "$batch_size" \
  "$model" \
  "train/lines/syms.txt" \
  "data/lists/lines/$partition/tr_h$height.lst" \
  "data/lang/char/lines/$partition/tr.txt" \
  /dev/null "${model/.t7/.prior}"
}

# Compute log-likelihoods from the network.
for p in va te; do
  lines_ark="decode/lkh/$partition/lines/${p}_${model_name}_ps${prior_scale}.ark";
  forms_ark="decode/lkh/$partition/forms/${p}_${model_name}_ps${prior_scale}.ark";
  # LINE log-likelihoods
  [ "$overwrite_likelihoods" = false -a -s "$lines_ark" ] || {
    overwrite_word_decode=true;
    overwrite_char_decode=true;
    ../../laia-netout \
      --batch_size "$batch_size" \
      --prior "${model/.t7/.prior}" \
      --prior_alpha "$prior_scale" \
      "$model" "data/lists/lines/$partition/${p}_h${height}.lst" /dev/stdout |
    copy-matrix ark:- "ark:$lines_ark";
  }
  # FORM log-likelihoods
  [ "$overwrite_likelihoods" = false -a -s "$forms_ark" ] || {
    overwrite_word_decode=true;
    overwrite_char_decode=true;
    ./utils/join_lines_arks.sh "train/lines/syms.txt" "$lines_ark" "$forms_ark";
  }
done;

# Build lexicon from the boundaries files.
lexiconp=data/lang/word/lexiconp.txt;
[ "$overwrite_lexicon" = false -a -s "$lexiconp" ] || {
  overwrite_word_fst=true;
  ./utils/build_word_lexicon.sh \
    data/lang/word/forms/aachen/tr_boundaries.txt \
    data/lang/word/external/*_boundaries.txt > "$lexiconp"
} ||
{ echo "ERROR: Creating file \"$lexiconp\"!" >&2 && exit 1; }


: <<EOF
# Build word-level language model WITHOUT the unknown token
[ "$overwrite_word_lm" = true ] && overwrite_word_fst=true;
./utils/build_word_lm.sh \
  --order "$word_order" --voc_size "$voc_size" \
  --unknown false --srilm_options "-kndiscount -interpolate" \
  --overwrite "$overwrite_word_lm" \
  train/lines/syms.txt \
  data/lang/word/forms/aachen/tr_{tokenized,boundaries}.txt \
  data/lang/word/forms/aachen/va_{tokenized,boundaries}.txt \
  data/lang/word/forms/aachen/te_{tokenized,boundaries}.txt \
  data/lang/word/external/lob_excludealltestsets_{tokenized,boundaries}.txt \
  data/lang/word/external/brown_{tokenized,boundaries}.txt \
  data/lang/word/external/wellington_{tokenized,boundaries}.txt \
  "decode/lm/$partition/word_lm";

# Build char-level language model for full forms.
[ "$overwrite_char_lm" = true ] && overwrite_char_fst=true;
./utils/build_char_lm.sh  \
  --order "$char_order" \
  --srilm_options "-wbdiscount -interpolate" \
  --overwrite "$overwrite_char_lm" \
  "data/lang/char/forms/$partition/tr.txt" \
  "data/lang/char/forms/$partition/va.txt" \
  "data/lang/char/forms/$partition/te.txt" \
  data/lang/char/external/lob_excludealltestsets.txt \
  data/lang/char/external/brown.txt \
  data/lang/char/external/wellington.txt \
  "decode/lm/$partition/char_lm";

# Build transducers for the word-based model.
[ "$overwrite_word_fst" = true ] && overwrite_word_decode=true;
./utils/build_word_fsts.sh --overwrite "$overwrite_word_fst" \
  train/lines/syms.txt data/lang/word/lexiconp.txt \
  "decode/lm/$partition/word_lm/interpolation-${word_order}gram-${voc_size}.arpa.gz" \
  "decode/lm/$partition/word_fst-${word_order}gram-${voc_size}";

# Build transducers for the char-based model.
[ "$overwrite_char_fst" = true ] && overwrite_char_decode=true;
./utils/build_char_fsts.sh --overwrite "$overwrite_char_fst" \
  train/lines/syms.txt \
  "decode/lm/$partition/char_lm/interpolation-${char_order}gram.arpa.gz" \
  "decode/lm/$partition/char_fst-${char_order}gram";
EOF

# Utility function to compute the errors
function compute_errors () {
  [ $# -ne 2 ] && echo "Usage: compute_errors forms_char forms_word" >&2 && return 1;
  ref_char="data/lang/char/forms/$partition/$p.txt";
  ref_word="data/lang/word/forms/$partition/$p.txt";
  # Some checks
  nc=( $(wc -l "$ref_char" "$1" | head -n2 | awk '{print $1}') );
  nw=( $(wc -l "$ref_word" "$2" | head -n2 | awk '{print $1}') );
  [[ "${nc[0]}" -eq "${nc[1]}" ]] || echo "HOLA";
  echo "WARNING: The number of reference forms does not match your char-level hypothesis (${nc[0]} vs. ${nc[1]})" >&2
  [[ "${nw[0]}" -eq "${nw[1]}" ]] ||
  echo "WARNING: The number of reference forms does not match your word-level hypothesis (${nw[0]} vs. ${nw[1]})" >&2;
  # Compute error rates.
  if [ $hasR -eq 1 ]; then
    # Compute CER and WER with Confidence Intervals using R
    tmpf="$(mktemp)";
    ./utils/compute-errors.py "data/lang/char/forms/$partition/$p.txt" "$1" > "$tmpf";
    ./utils/compute-confidence.R "$tmpf" |
    awk -v p="$p" '$1 == "%ERR"{ printf("%CER forms %s: %.2f %s %s %s\n", p, $2, $3, $4, $5); }';
    ./utils/compute-errors.py "data/lang/word/forms/$partition/$p.txt" "$2" > "$tmpf";
    ./utils/compute-confidence.R "$tmpf" |
    awk -v p="$p" '$1 == "%ERR"{ printf("%WER forms %s: %.2f %s %s %s\n", p, $2, $3, $4, $5); }';
    rm -f "$tmpf";
  elif [ $hasComputeWer -eq 1 ]; then
    # Compute CER and WER using Kaldi's compute-wer
    compute-wer --text --mode=strict \
      "ark:data/lang/char/forms/$partition/$p.txt" "ark:$1" \
      2>/dev/null |
    awk -v p="$p" '$1 == "%WER"{ printf("%CER forms %s: %.2f\n", p, $2); }';
    compute-wer --text --mode=strict \
      "ark:data/lang/word/forms/$partition/$p.txt" "ark:$2" \
      2>/dev/null |
    awk -v p="$p" '$1 == "%WER"{ printf("%WER forms %s: %.2f\n", p, $2); }';
  fi;
  return 0;
}

# Perform decoding and evaluation using different models.
m_desc=(
  "Word-level ${word_order}gram LM with ${voc_size} tokens"
  "Char-level ${char_order}gram LM"
);
fst_dir=(
  "decode/lm/$partition/word_fst-${word_order}gram-${voc_size}"
  "decode/lm/$partition/char_fst-${char_order}gram"
);
forms_char=(
  "decode/lm/$partition/char/${p}_${model_name}_ps${prior_scale}_${word_order}gram-${voc_size}.txt"
  "decode/lm/$partition/char/${p}_${model_name}_ps${prior_scale}_${char_order}gram.txt"
);
forms_word=(
  "decode/lm/$partition/word/${p}_${model_name}_ps${prior_scale}_${word_order}gram-${voc_size}.txt"
  "decode/lm/$partition/word/${p}_${model_name}_ps${prior_scale}_${char_order}gram.txt"
);
tmpd=(
  "${fst_dir[0]}/decode_workdir"
  "${fst_dir[1]}/decode_workdir"
);
overwrite_decode=(
  "$overwrite_word_decode"
  "$overwrite_char_decode"
);
asf=( $acoustic_scales );
lkh_dir="decode/lkh/$partition/forms";
for m in $(seq 1 ${#fst_dir[@]}); do
  for p in va te; do
    # Launch decoding.
    [ "${overwrite_decode[m-1]}" = false -a -s "${forms_char[m-1]}" ] || {
      # First, remove old transcripts, if any.
      rm -f "${forms_char[m-1]}" "${forms_word[m-1]}";
      ./utils/decode_lazy.sh \
        --acoustic_scale "${asf[m-1]}" \
        --beam "$beam" \
        --max_active "$max_active" \
        --num_tasks 200 \
        --overwrite "${overwrite_decode[m-1]}" \
        --qsub_opts "$qsub_opts" \
        "${fst_dir[m-1]}"/{model,HCL.fst,G.fst} \
        "$lkh_dir/${p}_${model_name}_ps0.2.scp" \
        "${tmpd[m-1]}/$p"
    } ||
    {
      echo "ERROR: Decoding failed, check logs in ${tmpd[m-1]}/$p" >&2 &&
      exit 1;
    }
  done;
done;

# If jobs are running on qsub, we must wait for them. Come back later.
[ -n "$qsub_opts" ] &&
echo "WARNING: qsub is running, execute this script again when all jobs" \
  "are done. Jobs are writting in: ${tmpd[@]}." &&
exit 0;

for m in "$(seq 1 ${#fst_dir[@]})"; do
  echo "${m_desc[m-1]}";
  for p in va te; do
    mkdir -p "$(dirname "${forms_char[m-1]}")";
    mkdir -p "$(dirname "${forms_word[m-1]}")";
    # Obtain char-level transcript for the forms.
    # The character sequence is produced by going through the HMM sequences
    # and then removing the dummy HMM boundaries (inc. whitespaces).
    [ "${overwrite_decode[m-1]}" = false -a -s "${forms_char[m-1]}" ] ||
    for f in "${tmpd[m-1]}"/align.*.of.*.ark.gz; do
      ali-to-phones "${fst_dir[m-1]}/model" "ark:zcat $f|" ark,t:- \
        2> /dev/null;
    done |
    ./utils/int2sym.pl -f 2- "${fst_dir[m-1]}/chars.txt" |
    ./utils/remove_transcript_dummy_boundaries.sh > "${forms_char[m-1]}";
    # Obtain the word-level transcript for the forms.
    # We just put together all characters that are not <space> to form words.
    [ "${overwrite_decode[m-1]}" = false -a -s "${forms_word[m-1]}" ] ||
    ./utils/remove_transcript_dummy_boundaries.sh --to-words \
      "${forms_char[m-1]}" > "${forms_word[m-1]}";
    # Compute errors
    compute_errors "${forms_char[m-1]}" "${forms_word[m-1]}";
  done;
done;

exit 0;
