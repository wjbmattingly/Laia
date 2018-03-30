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

acoustic_scale=1.9;
batch_size=16;
beam=65;
height=128;
max_active=2147483647;
overwrite=false;
prior_scale=0.2;
qsub_opts="";
voc_size=10000;
word_order=4;
help_message="
Usage: ${0##*/} [options] model

Options:
  --acoustic_scale : (type = float, default = $acoustic_scale)
                     Scaling factor for acoustic likelihoods (i.e. neural
                     network outputs).
  --batch_size     : (type = integer, default = $batch_size)
                     Batch size used to compute the network output.
  --beam           : (type = float, default = $beam)
                     Decoding beam.  Larger->slower, more accurate.
  --height         : (type = integer, default = $height)
                     Use images rescaled to this height.
  --max_active     : (type = integer, default = $max_active)
                     Max. number of tokens during Viterbi decoding
                     (a.k.a. histogram prunning).
  --overwrite      : (type = boolean, default = $overwrite)
                     Overwrite previously created files.
  --prior_scale    : (type = float, default = $prior_scale)
                     Scaling factor for the label priors used to compute the
                     pseudo-likelihoods.
  --qsub_opts      : (type = string, default = \"$qsub_opts\")
                     If any option is given, will parallelize the decoding
                     using qsub. THIS IS HIGHLY RECOMMENDED.
  --voc_size       : (type = integer, default = $voc_size)
                     Number of tokens in the word LM.
  --word_order     : (type = integer, default = $word_order)
                     Order of the n-gram word LM.
";
source utils/parse_options.inc.sh || exit 1;
[ $# -ne 1 ] && echo "$help_message" >&2 && exit 1;
model="$1";
model_name="$(basename "$1" .t7)";

# Check required files
for f in  "data/lists/te_h$height.lst" \
          "data/lists/tr_h$height.lst" \
	  "data/lists/va_h$height.lst" \
	  "data/lang/lines/char/tr.txt" \
  	  "data/lang/forms/char/te.txt" \
	  "data/lang/forms/char/va.txt" \
	  "data/lang/forms/word/te.txt" \
	  "data/lang/forms/word/va.txt" \
          "data/lang/forms/word/te_boundaries.txt" \
          "data/lang/forms/word/te_tokenized.txt" \
          "data/lang/forms/word/tr_boundaries.txt" \
          "data/lang/forms/word/tr_tokenized.txt" \
          "data/lang/forms/word/va_boundaries.txt" \
          "data/lang/forms/word/va_tokenized.txt" \
          "train/syms.txt" \
          "$model"; do
  [ ! -s "$f" ] && echo "ERROR: File \"$f\" was not found!" >&2 && exit 1;
done;

for f in decode-lazylm-faster-mapped ali-to-phones; do
  which "$f" &> /dev/null ||
  { echo "ERROR: Program \"$f\" is not in your PATH!" >&2 && exit 1; }
done;

hasR=0;
if which Rscript &> /dev/null; then hasR=1; fi;
hasComputeWer=0;
if which compute-wer &> /dev/null; then hasComputeWer=1; fi;

[ $hasR -ne 0 -o $hasComputeWer -ne 0 ] ||
echo "WARNING: Neither Rscript or compute-wer were found," \
  "CER/WER won't be computed!" >&2;

mkdir -p decode/lm/{char,word} decode/lkh/{lines,forms};

# Compute label priors
priors="$(dirname "$model")/${model_name}.prior";
[ "$overwrite" = false -a -s "$priors" ] ||
../../laia-force-align \
  --batch_size "$batch_size" \
  "$model" "train/syms.txt" \
  "data/lists/tr_h$height.lst" \
  "data/lang/lines/char/tr.txt" \
  /dev/null "$priors";

# Compute log-likelihoods from the network.
for p in va te; do
  lines_ark="decode/lkh/lines/${p}_${model_name}_ps${prior_scale}.ark";
  forms_ark="decode/lkh/forms/${p}_${model_name}_ps${prior_scale}.ark";
  # LINE log-likelihoods
  [ "$overwrite" = false -a -s "$lines_ark" ] ||
  ../../laia-netout \
    --batch_size "$batch_size" \
    --prior "$priors" \
    --prior_alpha "$prior_scale" \
    "$model" "data/lists/${p}_h$height.lst" \
    /dev/stdout |
  copy-matrix ark:- "ark:$lines_ark";
  # FORM log-likelihoods
  [ "$overwrite" = false -a -s "$forms_ark" ] ||
  ./utils/join_lines_arks.sh --add_wspace_border true \
    "train/syms.txt" "$lines_ark" "$forms_ark";
done;

# Build lexicon from the boundaries file.
lexiconp=data/lang/forms/word/lexiconp.txt;
[ "$overwrite" = false -a -s "$lexiconp" ] ||
./utils/build_word_lexicon.sh \
  data/lang/forms/word/tr_boundaries.txt > "$lexiconp" ||
{ echo "ERROR: Creating file \"$lexiconp\"!" >&2 && exit 1; }

# Build word-level language model WITHOUT the unknown token
./utils/build_word_lm.sh --order "$word_order" --voc_size "$voc_size" \
  --unknown false --srilm_options "-kndiscount -interpolate" \
  --overwrite "$overwrite" \
  train/syms.txt \
  data/lang/forms/word/tr_{tokenized,boundaries}.txt \
  data/lang/forms/word/va_{tokenized,boundaries}.txt \
  data/lang/forms/word/te_{tokenized,boundaries}.txt \
  decode/lm/word_lm;

# Build decoding FSTs for the word-level language model
./utils/build_word_fsts.sh \
  train/syms.txt "$lexiconp" \
  "decode/lm/word_lm/tr_tokenized-${word_order}gram-${voc_size}.arpa.gz" \
  "decode/lm/word_fst-${word_order}gram-${voc_size}";

# Decode.
tmpd="decode/lm/word_fst-${word_order}gram-${voc_size}/decode_workdir";
qsub_jobs=();
for p in va te; do
  forms_char="decode/lm/char/${p}_${model_name}.txt";
  forms_word="decode/lm/word/${p}_${model_name}.txt";
  # Launch decoding.
  [ "${overwrite}" = false -a -s "${forms_char}" ] || {
    # First, remove old transcripts, if any.
    rm -f "${forms_char}" "${forms_word}";
    qsub_jobs+=( $(./utils/decode_lazy.sh \
      --acoustic_scale "${acoustic_scale}" \
      --beam "$beam" \
      --max_active "$max_active" \
      --num_tasks 350 \
      --overwrite "$overwrite" \
      --qsub_opts "$qsub_opts" \
      "decode/lm/word_fst-${word_order}gram-${voc_size}"/{model,HCL.fst,G.fst} \
      "decode/lkh/forms/${p}_${model_name}_ps${prior_scale}.scp" \
      "$tmpd/$p") );
  } ||
  {
    echo "ERROR: Decoding failed, check logs in ${tmpd}/$p" >&2 &&
    exit 1;
  }
done;

# If jobs are running on qsub, we must wait for them. Come back later.
[ "${#qsub_jobs[@]}" -gt 0 ] &&
echo "WARNING: qsub is running, execute this again when all jobs are done." &&
exit 0;


# Utility function to compute the errors
function compute_errors () {
  [ $# -ne 2 ] && echo "Usage: compute_errors forms_char forms_word" >&2 &&
  return 1;
  ref_char="data/lang/forms/char/$p.txt";
  ref_word="data/lang/forms/word/$p.txt";
  # Some checks
  nc=( $(wc -l "$ref_char" "$1" | head -n2 | gawk '{print $1}') );
  nw=( $(wc -l "$ref_word" "$2" | head -n2 | gawk '{print $1}') );
  [[ "${nc[0]}" -eq "${nc[1]}" ]] ||
  echo "WARNING: The number of reference forms does not match your char" \
    "hypothesis (${nc[0]} vs. ${nc[1]})" >&2
  [[ "${nw[0]}" -eq "${nw[1]}" ]] ||
  echo "WARNING: The number of reference forms does not match your word" \
    "hypothesis (${nw[0]} vs. ${nw[1]})" >&2;
  if [ $hasR -eq 1 ]; then
    # Compute CER and WER with Confidence Intervals using R
    ./utils/compute-errors.py "$ref_char" "$forms_char" > "$tmpf";
    ./utils/compute-confidence.R "$tmpf" |
    gawk -v p="$p" '$1 == "%ERR"{
      printf("%CER forms %s: %.2f %s %s %s\n", p, $2, $3, $4, $5); }';
    ./utils/compute-errors.py "$ref_word" "$forms_word" > "$tmpf";
    ./utils/compute-confidence.R "$tmpf" |
    gawk -v p="$p" '$1 == "%ERR"{
      printf("%WER forms %s: %.2f %s %s %s\n", p, $2, $3, $4, $5); }';
  elif [ $hasComputeWer -eq 1 ]; then
    # Compute CER and WER using Kaldi's compute-wer
    compute-wer --text --mode=strict "ark:$ref_char" "ark:$lines_char" \
      2>/dev/null |
    gawk -v p="$p" '$1 == "%WER"{ printf("%CER forms %s: %.2f\n", p, $2); }';
    compute-wer --text --mode=strict "ark:$ref_word" "ark:$lines_word" \
      2>/dev/null |
    gawk -v p="$p" '$1 == "%WER"{ printf("%WER forms %s: %.2f\n", p, $2); }';
  fi;
  rm -f "$tmpf";
  return 0;
}

# Merge decode results & compute errors.
tmpf="$(mktemp)";
for p in va te; do
  forms_ark="decode/lkh/forms/${p}_${model_name}_ps${prior_scale}.ark";
  forms_char="decode/lm/char/${p}_${model_name}.txt";
  forms_word="decode/lm/word/${p}_${model_name}.txt";
  # Obtain char-level transcript for the forms.
  # The character sequence is produced by going through the HMM sequences and
  # then removing some of the dummy HMM boundaries.
  [ "$overwrite" = false -a -s "$forms_char" ] ||
  for f in "${tmpd}/$p"/align.*.of.*.ark.gz; do
    ali-to-phones \
      "decode/lm/word_fst-${word_order}gram-${voc_size}/model" "ark:zcat $f|" \
      ark,t:- 2> /dev/null
  done |
  ./utils/int2sym.pl -f 2- \
    "decode/lm/word_fst-${word_order}gram-${voc_size}/chars.txt" |
  ./steps/remove_transcript_dummy_boundaries.sh > "$forms_char";
  # Obtain the word-level transcript for the forms.
  # We just put together all characters that are not <space> to form words.
  [ "$overwrite" = false -a -s "$forms_word" ] ||
  ./steps/remove_transcript_dummy_boundaries.sh \
    --to-words "$forms_char" > "$forms_word";
  # Compute errors
  compute_errors "$forms_char" "$forms_word";
done;
rm -f "$tmpf";
