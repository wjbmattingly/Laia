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

acoustic_scales="1.79 1.30";
batch_size=16;
beam=65;
char_order=-1;
height=128;
max_active=5000000;
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
  --acoustic_scales : (type = string, default \"$acoustic_scales\")
                      List of acoustic scale factors, separated by spaces.
                      The first is used with a word LM and the second with
                      a character LM.
  --batch_size      : (type = integer, default = $batch_size)
                      Batch size for Laia.
  --beam            : (type = float, default = $beam)
                      Decoding beam.
  --char_order      : (type = integer, default = $char_order)
                      Order of the n-gram character LM. Use -1 to disable this.
  --height          : (type = integer, default = $height)
                      Use images rescaled to this height.
  --max_active      : (type = integer, default = $max_active)
                      Max. number of tokens during Viterbi decoding
                      (a.k.a. histogram prunning).
  --overwrite       : (type = boolean, default = $overwrite)
                      Overwrite ALL previous stages.
  --partition       : (type = string, default = \"$partition\")
                      Select \"aachen\" or \"kws\".
  --prior_scale     : (type = float, default = $prior_scale)
                      Use this scale factor on the label priors to convert the
                      softmax output of the neural network into likelihoods.
  --qsub_opts       : (type = string, default = \"$qsub_opts\")
                      If any option is given, will parallelize the decoding
                      using qsub. THIS IS HIGHLY RECOMMENDED.
  --voc_size        : (type = integer, default = $voc_size)
                      Vocabulary size for the word n-gram LM.
  --word_order      : (type = integer, default = $word_order)
                      Order of the n-gram word LM. Use -1 to disable this.
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
    data/lang/word/forms/$partition/tr_boundaries.txt \
    data/lang/word/external/*_boundaries.txt > "$lexiconp"
} ||
{ echo "ERROR: Creating file \"$lexiconp\"!" >&2 && exit 1; }

# Build word-level language model WITHOUT the unknown token
[ "$overwrite_word_lm" = true ] && overwrite_word_fst=true;
[ "$word_order" -gt 0 ] &&
./utils/build_word_lm.sh \
  --order "$word_order" --voc_size "$voc_size" \
  --unknown false --srilm_options "-kndiscount -interpolate" \
  --overwrite "$overwrite_word_lm" \
  train/lines/syms.txt \
  data/lang/word/forms/$partition/tr_{tokenized,boundaries}.txt \
  data/lang/word/forms/$partition/va_{tokenized,boundaries}.txt \
  data/lang/word/forms/$partition/te_{tokenized,boundaries}.txt \
  data/lang/word/external/lob_excludealltestsets_{tokenized,boundaries}.txt \
  data/lang/word/external/brown_{tokenized,boundaries}.txt \
  data/lang/word/external/wellington_{tokenized,boundaries}.txt \
  "decode/lm/$partition/word_lm";

# Build transducers for the word-based model.
[ "$overwrite_word_fst" = true ] && overwrite_word_decode=true;
[ "$word_order" -gt 0 ] &&
./utils/build_word_fsts.sh --overwrite "$overwrite_word_fst" \
  train/lines/syms.txt data/lang/word/lexiconp.txt \
  "decode/lm/$partition/word_lm/interpolation-${word_order}gram-${voc_size}.arpa.gz" \
  "decode/lm/$partition/word_fst-${word_order}gram-${voc_size}";

# Build char-level language model for full forms.
[ "$overwrite_char_lm" = true ] && overwrite_char_fst=true;
[ "$char_order" -gt 0 ] &&
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

# Build transducers for the char-based model.
[ "$overwrite_char_fst" = true ] && overwrite_char_decode=true;
[ "$char_order" -gt 0 ] &&
./utils/build_char_fsts.sh --overwrite "$overwrite_char_fst" \
  train/lines/syms.txt \
  "decode/lm/$partition/char_lm/interpolation-${char_order}gram.arpa.gz" \
  "decode/lm/$partition/char_fst-${char_order}gram";

# Utility function to compute the errors
function compute_errors () {
  [ $# -ne 2 ] && echo "Usage: compute_errors forms_char forms_word" >&2 && return 1;
  ref_char="data/lang/char/forms/$partition/$p.txt";
  ref_word="data/lang/word/forms/$partition/$p.txt";
  # Some checks
  nc=( $(wc -l "$ref_char" "$1" | head -n2 | awk '{print $1}') );
  nw=( $(wc -l "$ref_word" "$2" | head -n2 | awk '{print $1}') );
  [[ "${nc[0]}" -eq "${nc[1]}" ]] ||
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
m_desc=();
fst_dir=();
txt_sfx=();
asf=();
overwrite_decode=();
[ "$word_order" -gt 0 ] && {
  m_desc+=("Word-level ${word_order}gram LM with ${voc_size} tokens");
  fst_dir+=("decode/lm/$partition/word_fst-${word_order}gram-${voc_size}");
  txt_sfx+=("${model_name}_ps${prior_scale}_${word_order}gram-${voc_size}.txt");
  asf+=("$acoustic_scale_word");
  overwrite_decode+=("$overwrite_word_decode");
}
[ "$char_order" -gt 0 ] && {
  m_desc+=("Char-level ${char_order}gram LM");
  fst_dir+=("decode/lm/$partition/char_fst-${char_order}gram");
  txt_sfx+=("${model_name}_ps${prior_scale}_${char_order}gram.txt");
  asf+=("$acoustic_scale_char");
  overwrite_decode+=("$overwrite_char_decode");
}
asf=( $acoustic_scales );
lkh_dir="decode/lkh/$partition/forms";
qsub_jobs+=();
for m in $(seq 1 ${#fst_dir[@]}); do
  tmpd="${fst_dir[m-1]}/decode_workdir";
  for p in va te; do
    forms_char="decode/lm/$partition/char/${p}_${txt_sfx[m-1]}";
    forms_word="decode/lm/$partition/word/${p}_${txt_sfx[m-1]}";
    # Launch decoding.
    [ "${overwrite_decode[m-1]}" = false -a -s "${forms_char}" ] || {
      # First, remove old transcripts, if any.
      rm -f "${forms_char}" "${forms_word}";
      qsub_jobs+=( $(./utils/decode_lazy.sh \
        --acoustic_scale "${asf[m-1]}" \
        --beam "$beam" \
        --max_active "$max_active" \
        --num_tasks 350 \
        --overwrite "${overwrite_decode[m-1]}" \
        --qsub_opts "$qsub_opts" \
        "${fst_dir[m-1]}"/{model,HCL.fst,G.fst} \
        "$lkh_dir/${p}_${model_name}_ps0.2.scp" \
        "${tmpd}/$p") );
    } ||
    {
      echo "ERROR: Decoding failed, check logs in ${tmpd}/$p" >&2 &&
      exit 1;
    }
  done;
done;

# If jobs are running on qsub, we must wait for them. Come back later.
[ "${#qsub_jobs[@]}" -gt 0 ] &&
echo "WARNING: qsub is running, execute this again when all jobs are done." &&
exit 0;

for m in $(seq 1 ${#fst_dir[@]}); do
  echo "${m_desc[m-1]}";

  for p in va te; do
    forms_char="decode/lm/$partition/char/${p}_${txt_sfx[m-1]}";
    forms_word="decode/lm/$partition/word/${p}_${txt_sfx[m-1]}";
    mkdir -p "$(dirname "${forms_char}")";
    mkdir -p "$(dirname "${forms_word}")";
    # Obtain char-level transcript for the forms.
    # The character sequence is produced by going through the HMM sequences
    # and then removing the dummy HMM boundaries (inc. whitespaces).
    [ "${overwrite_decode[m-1]}" = false -a -s "${forms_char}" ] ||
    for f in "${tmpd}/$p"/align.*.of.*.ark.gz; do
      ali-to-phones "${fst_dir[m-1]}/model" "ark:zcat $f|" ark,t:- 2> /dev/null
    done |
    ./utils/int2sym.pl -f 2- "${fst_dir[m-1]}/chars.txt" |
    ./utils/remove_transcript_dummy_boundaries.sh > "${forms_char}";
    # Obtain the word-level transcript for the forms.
    # We just put together all characters that are not <space> to form words.
    [ "${overwrite_decode[m-1]}" = false -a -s "${forms_word}" ] ||
    ./utils/remove_transcript_dummy_boundaries.sh --to-words \
      "${forms_char}" > "${forms_word}";
    # Compute errors
    compute_errors "${forms_char}" "${forms_word}";
  done;
done;

exit 0;
