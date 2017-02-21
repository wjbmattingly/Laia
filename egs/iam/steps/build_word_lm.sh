#!/bin/bash
export LC_NUMERIC=C;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/steps" = "$SDIR" ] || {
    echo "Run the script from \"$(dirname $SDIR)\"!" >&2 && exit 1;
}
[ ! -f "$(pwd)/utils/parse_options.inc.sh" ] && \
    echo "Missing $(pwd)/utils/parse_options.inc.sh file!" >&2 && exit 1;

add_train_voc=false;
order=3;
overwrite=false;
ptr=lines;
pva=lines;
voc_size=20000;
srilm_options="-kndiscount -interpolate -unk";
help_message="
Usage: ${0##*/} [options] charmap input_dir output_dir

Description:
  Build a word-level N-gram language model using SRILM. You can specify the
  order and the vocabulary size of the N-gram using --order and --voc_size
  options, as well as the type of discount and other SRILM options with
  --srilm_options.

Arguments:
  charmap         : File containing the mapping from characters to HMMs
                    (e.g. \"data/text/charmap\").
  input_dir       : Directory containing the tokenized files and the boundaries
                    files of all the text data (e.g. \"exper/htr/lang/word\").
  output_dir      : Output directory where the language models will be written
                    together with other auxiliary files
                    (e.g. \"exper/htr/lang/word/lm\").

Options:
  --add_train_voc : (type = boolean, default = $add_train_voc)
                    If true, add all words in the training set to the vocabulary
                    and the extend this with external words.
  --order         : (type = integer, default = $order)
                    Order of the n-gram language model.
  --overwrite     : (type = boolean, default = $overwrite)
                    Overwrite previously created files.
  --ptr           : (type = string, default = \"$ptr\")
                    Specify which partition of IAM training data is used.
                    Valid values = forms, lines, sentences.
  --pva           : (type = string, default = \"$pva\")
                    Specify which partition of IAM validation data is used.
                    Valid values = forms, lines, sentences.
  --srilm_options : (type = string, default = \"$srilm_options\")
                    Use SRILM's ngram-count with these options.
  --voc_size      : (type = integer, default = $voc_size)
                    Keep only this number of words.
";
source "$(pwd)/utils/parse_options.inc.sh" || exit 1;
[ $# -ne 3 ] && echo "$help_message" >&2 && exit 1;

cmap="$1";
idir="$2";
odir="$3";

mkdir -p "$odir";

vocf="$odir/voc-${voc_size}";
va_txt="$idir/iam/$pva/va_tokenized.txt";

# Check required files
for f in "$idir/iam/$ptr/tr_tokenized.txt" \
  "$idir/iam/$ptr/tr_boundaries.txt" \
  "$idir/brown_tokenized.txt" \
  "$idir/lob_excludealltestsets_tokenized.txt" \
  "$idir/wellington_tokenized.txt" \
  "$idir/brown_boundaries.txt" \
  "$idir/lob_excludealltestsets_boundaries.txt" \
  "$idir/wellington_boundaries.txt" \
  "$va_txt" \
  "$cmap" ; do
  [ ! -s "$f" ] && echo "File \"$f\" was not found!" >&2 && exit 1;
done;

# Get total counts of each token across the different corpora
tmpf="$(mktemp)";
awk '{ C[$1] += $2; }END{ for (w in C) print w, C[w]; }' \
  "$idir/brown_boundaries.txt" \
  "$idir/lob_excludealltestsets_boundaries.txt" \
  "$idir/wellington_boundaries.txt" |
awk -v CMF="$cmap" '
BEGIN{
  while((getline < CMF) > 0) { c=$1; $1=""; gsub(/^\ +/, ""); M[c]=$0; }
}{
  for (i=1;i<=length($1);++i) { if (!(substr($1, i, 1) in M)) { N++; next; } }
  print;
}END{
  print N" tokens ignored due to out-of-training symbols!" > "/dev/stderr";
}' |
awk '{ C[$1] = C[$1] + $2; }END{ for (w in C) print w, C[w]; }' |
sort -nrk2 |
if [ "$add_train_voc" = true ]; then
  # With --add_train_voc=true we include all tokens in IAM train set
  # and extend this list with tokens from the external datasets until
  # a maximum of --voc_size tokens are used.
  awk -v TBF="$idir/iam/$ptr/tr_boundaries.txt" '
BEGIN{
  while ((getline < TBF) > 0) { TV[$1]=1; print $1; }
}{
  if (($1 in TV)) next;
  print $1;
}' | head -n "$voc_size";
else
  # With --add_train_voc=false we sort all tokens by absolute frequency and
  # just use the most --voc_size frequent.
  head -n "$voc_size" | awk '{print $1}';
fi | sort -V > "$tmpf";
if [ "$overwrite" = true ] || ! cmp -s "$tmpf" "$vocf"; then
  mv "$tmpf" "$vocf";
fi;


# Train N-gram on the IAM data
mkdir -p "$odir/iam/$ptr";
outf="$odir/iam/$ptr/tr-${order}gram-${voc_size}.arpa.gz";
info="$odir/iam/$ptr/tr-${order}gram-${voc_size}.info";
[[ "$overwrite" = false && -s "$outf" &&
    ( ! "$outf" -ot "$idir/iam/$ptr/tr_tokenized.txt" ) &&
    ( ! "$outf" -ot "$vocf" ) ]] ||
awk '{$1=""; print;}' "$idir/iam/$ptr/tr_tokenized.txt" |
ngram-count -order "$order" -vocab "$vocf" $srilm_options -text - -lm - |
gzip -9 -c > "$outf" ||
exit 1;
# Compute detailed perplexity
[[ "$overwrite" = false && -s "$info" && ( ! "$info" -ot "$outf" ) ]] ||
awk '{$1=""; print;}' "$va_txt" |
ngram -order "$order" -debug 2 -ppl - -lm <(zcat "$outf") \
  &> "$info" ||
exit 1;


# Train N-gram on the external data
for c in brown lob_excludealltestsets wellington; do
  # Train n-gram
  outf="$odir/$c-${order}gram-${voc_size}.arpa.gz";
  info="$odir/$c-${order}gram-${voc_size}.info";
  [[ "$overwrite" = false && -s "$outf" &&
      ( ! "$outf" -ot "$idir/${c}_tokenized.txt" ) &&
      ( ! "$outf" -ot "$vocf" ) ]] ||
  cat "$idir/${c}_tokenized.txt" |
  ngram-count -order "$order" -vocab "$vocf" $srilm_options -text - -lm - |
  gzip -9 -c > "$outf" ||
  exit 1;
  # Compute detailed perplexity
  [[ "$overwrite" = false && -s "$info" && ( ! "$info" -ot "$outf" ) ]] ||
  awk '{$1=""; print;}' "$va_txt" |
  ngram -order "$order" -debug 2 -ppl - -lm <(zcat "$outf") \
    &> "$info" ||
  exit 1;
done;


# Compute interpolation weights
outf="$odir/interpolation-${order}gram-${voc_size}.mix";
[[ "$overwrite" = false && -s "$outf" &&
    ( ! "$outf" -ot "$odir/iam/lines/tr-${order}gram-${voc_size}.info" ) &&
    ( ! "$outf" -ot "$odir/brown-${order}gram-${voc_size}.info" ) &&
    ( ! "$outf" -ot "$odir/lob_excludealltestsets-${order}gram-${voc_size}.info" ) &&
    ( ! "$outf" -ot "$odir/wellington-${order}gram-${voc_size}.info" ) ]] ||
compute-best-mix \
  "$odir/iam/lines/tr-${order}gram-${voc_size}.info" \
  "$odir/brown-${order}gram-${voc_size}.info" \
  "$odir/lob_excludealltestsets-${order}gram-${voc_size}.info" \
  "$odir/wellington-${order}gram-${voc_size}.info" \
  &> "$outf" ||
exit 1;


# Interpolate n-grams
outf="$odir/interpolation-${order}gram-${voc_size}.arpa.gz";
lambdas=( $(grep "best lambda" \
  "$odir/interpolation-${order}gram-${voc_size}.mix" | \
  awk -F\( '{print $2}' | tr -d \)) );
[[ "$overwrite" = false && -s "$outf" &&
    ( ! "$outf" -ot "$odir/interpolation-${order}gram-${voc_size}.mix" ) ]] ||
ngram -order "${order}" \
  -lm <(zcat "$odir/iam/lines/tr-${order}gram-${voc_size}.arpa.gz") \
  -mix-lm <(zcat "$odir/brown-${order}gram-${voc_size}.arpa.gz") \
  -mix-lm2 <(zcat "$odir/lob_excludealltestsets-${order}gram-${voc_size}.arpa.gz") \
  -mix-lm3 <(zcat "$odir/wellington-${order}gram-${voc_size}.arpa.gz") \
  -lambda "${lambdas[0]}" \
  -mix-lambda2 "${lambdas[2]}" \
  -mix-lambda3 "${lambdas[3]}" \
  -write-lm - 2> /dev/null |
gzip -9 -c > "$outf" ||
exit 1;


# Compute detailed perplexity of the interpolated model
ppl=();
oov=();
tok=();
for f in "$idir/iam/$ptr/tr_tokenized.txt"  \
  "$idir/iam/$pva/va_tokenized.txt" \
  "$idir/iam/$pva/te_tokenized.txt"; do
  ppl+=( $(awk '{$1=""; print;}' "$f" |
      ngram -order "$order" -ppl - -lm <(zcat "$outf") 2>&1 |
      tail -n1 | sed -r 's|^.+\bppl= ([0-9.]+)\b.+$|\1|g') );
  aux=( $(awk -v VF="$vocf" '
BEGIN{ N=0; OOV=0; while((getline < VF) > 0) V[$1]=1; }
{ for (i=2;i<=NF;++i) { ++N; if (!($i in V)) ++OOV; } }
END{ print OOV, N; }' "$f") );
  oov+=(${aux[0]});
  tok+=(${aux[1]});
done;


# Print statistics
cat <<EOF >&2
Word-level ${order}-gram with ${voc_size} tokens:
Train: ppl = ${ppl[0]}, oov = ${oov[0]}, %oov = $(echo "scale=2; 100 * ${oov[0]} / ${tok[0]}" | bc -l)
Valid: ppl = ${ppl[1]}, oov = ${oov[1]}, %oov = $(echo "scale=2; 100 * ${oov[1]} / ${tok[1]}" | bc -l)
Test:  ppl = ${ppl[2]}, oov = ${oov[2]}, %oov = $(echo "scale=2; 100 * ${oov[2]} / ${tok[2]}" | bc -l)
EOF

exit 0;
