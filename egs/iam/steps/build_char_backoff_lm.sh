#!/bin/bash
export LC_NUMERIC=C;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/steps" = "$SDIR" ] || {
    echo "Run the script from \"$(dirname $SDIR)\"!" >&2 && exit 1;
}
[ ! -f "$(pwd)/utils/parse_options.inc.sh" ] && \
    echo "Missing $(pwd)/utils/parse_options.inc.sh file!" >&2 && exit 1;

ctc="<ctc>";
eps="<eps>";
exclude_vocab="";
order=10;
overwrite=false;
partition=sentences/aachen;
srilm_options="-wbdiscount -interpolate";
wspace="<space>";
help_message="
Usage: ${0##*/} [options] output_dir

Description:
  Build a character-level N-gram language model using SRILM used to replace
  the <unk> tokens in a word-level languange model. You can specify the
  order using --order. If you want to exclude the vocabulary in your word-level
  language model to create this backoff model, use the --exclude_vocab option.
  Other SRILM options can be modified with --srilm_options.

Arguments:
  output_dir      : Output directory where the language models and other
                    files will be written (e.g. \"decode/lm_backoff\").

Options:
  --ctc           : (type = string, default = \"$ctc\")
                    Token representing the CTC blank symbol.
  --eps           : (type = string, default = \"$eps\")
                    Token representing the epsilon symbol.
  --exclude_vocab : (type = string, default = \"$exclude_vocab\")
                    Exclude the words in this file from the training data
                    (file contains one word per line).
  --order         : (type = int, default = $order)
                    Maximum order of N-grams to count.
  --overwrite     : (type = int, default = $overwrite)
                    Overwrite existing files from previous runs.
  --partition     : (type = string, default = \"$partition\")
                    Specify which partition of IAM is used.
                    Valid values = forms, lines, sentences.
  --srilm_options : (type = string, default = \"$srilm_options\")
                    Use SRILM's ngram-count with these options.
  --wspace        : (type = string, default \"$wspace\")
                    Token representing the whitespace character.
";
source utils/parse_options.inc.sh || exit 1;
[ $# -ne 1 ] && echo "$help_message" >&2 && exit 1;

odir="$1";
# Get "lines" or "sentences" from the full partition string (e.g. lines/aachen)
ptype="${partition%%/*}";
# Output directory
mkdir -p "$odir";
# Temporary directory
tmpd="$(mktemp -d)";

# Check that $1 is not newer than $2...$#.
function check_notnew () {
  for i in $(seq 2 $#); do [[ "${!i}" -ot "$1" ]] && return 1; done;
  return 0;
}

# Convert a lines of character-level transcripts sentences into words.
function sent2word () {
  sed "s|$wspace|\n|g" $@ | sed -r 's|^\ +||g;s|\ +$||g;s|\ +| |g';
}

# Exclude lines present in the file $exclude_vocab
function excludevoc () {
  awk -v VF="$exclude_vocab" '
BEGIN{ while((getline < VF) > 0) V[$0]=1; }
!($0 in V){ print; }' $@;
}

# Interpolate a list of .arpa.gz files.
function interpolate_arpa_files () {
  # Compute detailed perplexity on the validation data
  info_files=();
  for arpa in $@; do
    info="${arpa/.arpa.gz/.info}";
    [[ "$overwrite" = false && -s "$info" && ( ! "$info" -ot "$arpa" ) ]] ||
    sent2word "$tmpd/va" | excludevoc |
    ngram -order "$order" -debug 2 -ppl - -lm <(zcat "$arpa") &> "$info" ||
    { echo "ERROR: Creating file \"$info\"!" >&2 && exit 1; }
    info_files+=("$info");
  done;
  # Compute interpolation weights
  mixf="$odir/interpolation-${order}gram.mix";
  ( [[ "$overwrite" = false && -s "$mixf" ]] &&
    check_notnew "$mixf" "${info_files[@]}" ) ||
  compute-best-mix "${info_files[@]}" &> "$mixf" ||
  { echo "ERROR: Creating file \"$mixf\"!" >&2 && exit 1; }
  lambdas=( $(grep "best lambda" "$mixf" | awk -F\( '{print $2}' | tr -d \)) );
  # Interpolate language models.
  tmpfs=();
  args=();
  for i in $(seq 1 $#); do
    tmpfs+=( "$(mktemp)"  );
    zcat "${!i}" > "${tmpfs[${#tmpfs[@]} - 1]}";
    if [ $i -eq 1 ]; then
      args+=( -lm "${tmpfs[${#tmpfs[@]} - 1]}" -lambda "${lambdas[i - 1]}" );
    elif [ $i -eq 2 ]; then
      args+=( -mix-lm "${tmpfs[${#tmpfs[@]} - 1]}" );
    else
      args+=( "-mix-lm$[i - 1]" "${tmpfs[${#tmpfs[@]} - 1]}" \
              "-mix-lambda$[i - 1]" "${lambdas[i - 1]}" );
    fi;
  done;
  outf="$odir/interpolation-${order}gram.arpa.gz";
  [[ "$overwrite" = false && -s "$outf" && ( ! "$outf" -ot "$mixf" ) ]] ||
  ngram -order "${order}" "${args[@]}" -write-lm - |
  gzip -9 -c  > "$outf" ||
  { echo "ERROR: Creating file \"$outf\"!" >&2 && exit 1; }
  rm -f "${tmpfs[@]}";
  return 0;
}

# Obtain the sequence of characters from the given word vocabulary that must
# be excluded.
if [ -n "$exclude_vocab" ]; then
  [ ! -s "$exclude_vocab" ] &&
  echo "ERROR: File \"$exclude_vocab\" does not exist!" >&2 && exit 1;
  awk '{
    for (i=1;i<=length($0);++i) { printf("%s ", substr($0, i, 1)); }
    printf("\n");
  }' "$exclude_vocab" |
  sed -r 's|^\ +||g;s|\ +$||g;s|\ +| |g' > "$tmpd/exclude_vocab" ||
  exit 1;
  exclude_vocab="$tmpd/exclude_vocab";
else
  exclude_vocab="/dev/null";
fi;

# Check required files.
for f in "train/$ptype/syms.txt" \
         "data/lang/char/$partition/te.txt" \
         "data/lang/char/$partition/tr.txt" \
         "data/lang/char/$partition/va.txt" \
         "data/lang/char/external/brown.txt" \
         "data/lang/char/external/lob_excludealltestsets.txt" \
         "data/lang/char/external/wellington.txt"; do
  [ ! -s "$f" ] && echo "ERROR: File \"$f\" does not exist!" >&2 && exit 1;
done;

# Check required tools (SRILM)
for f in ngram-count ngram compute-best-mix; do
  which "$f" &> /dev/null ||
  { echo "ERROR: Program $f is not in your PATH!" >&2 && exit 1; }
done;

cut -d\  -f2- "data/lang/char/$partition/tr.txt" > "$tmpd/tr";
cut -d\  -f2- "data/lang/char/$partition/te.txt" > "$tmpd/te";
cut -d\  -f2- "data/lang/char/$partition/va.txt" > "$tmpd/va";
cut -d\  -f1  "train/$ptype/syms.txt" |
grep -v "$eps" | grep -v "$ctc" | grep -v "$wspace" > "$tmpd/voc";

arpa_files=();
for f in "$tmpd/tr" \
         "data/lang/char/external/brown.txt" \
         "data/lang/char/external/lob_excludealltestsets.txt" \
         "data/lang/char/external/wellington.txt"; do
  c="$(basename "$f")"; c="${c%.*}";
  arpa="$odir/${c}-${order}gram.arpa.gz";
  [[ "$overwrite" = false && -s "$arpa" ]] ||
  ngram-count -order "$order" -vocab "$tmpd/voc" $srilm_options \
    -text <(sent2word "$f" | excludevoc) -lm - 2> /dev/null |
  gzip -9 -c > "$arpa" ||
  { echo "ERROR: Creating file \"$arpa\"" >&2 && exit 1; }
  arpa_files+=("$arpa");
done;

# Interpolate all language models
interpolate_arpa_files "${arpa_files[@]}";
outf="$odir/interpolation-${order}gram.arpa.gz";

# Compute detailed perplexity of the interpolated model
ppl=();
oov=();
tok=();
for f in "$tmpd/tr" "$tmpd/va" "$tmpd/te"; do
  ppl+=( $(ngram -order "$order" -ppl <(sent2word "$f" | excludevoc) \
    -lm <(zcat "$outf") 2>&1 |
      tail -n1 | sed -r 's|^.+\bppl= ([0-9.]+)\b.+$|\1|g') );
  aux=( $(awk -v VF="$tmpd/voc" '
BEGIN{ N=0; OOV=0; while((getline < VF) > 0) V[$1]=1; }
{ for (i=1;i<=NF;++i) { ++N; if (!($i in V)) ++OOV; } }
END{ print OOV, N; }' <(sent2word "$f" | excludevoc)) );
  oov+=(${aux[0]});
  tok+=(${aux[1]});
done;

# Print statistics
cat <<EOF >&2
Character-level Backoff ${order}-gram:
Train: ppl = ${ppl[0]}, oov = ${oov[0]}, %oov = $(echo "scale=2; 100 * ${oov[0]} / ${tok[0]}" | bc -l)
Valid: ppl = ${ppl[1]}, oov = ${oov[1]}, %oov = $(echo "scale=2; 100 * ${oov[1]} / ${tok[1]}" | bc -l)
Test:  ppl = ${ppl[2]}, oov = ${oov[2]}, %oov = $(echo "scale=2; 100 * ${oov[2]} / ${tok[2]}" | bc -l)
EOF

# Remove temporary dir.
rm -rf "$tmpd";

exit 0;
