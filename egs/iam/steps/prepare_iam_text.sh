#!/bin/bash
set -e;

# Directory where the prepare.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/steps" != "$SDIR" ] && \
    echo "Please, run this script from the experiment top directory!" >&2 && \
    exit 1;
[ ! -f "$(pwd)/utils/parse_options.inc.sh" ] && \
    echo "Missing $(pwd)/utils/parse_options.inc.sh file!" >&2 && exit 1;

overwrite=false;
partition=lines;
wspace="<space>";
help_message="
Usage: ${0##*/} [options]

Options:
  --overwrite  : (type = boolean, default = $overwrite)
                 Overwrite previously created files.
  --partition  : (type = string, default = \"$partition\")
                 Select the \"lines\" or \"sentences\" partition. Note: Aachen
                 typically uses the sentences partition.
  --wspace     : (type = string, default \"$wspace\")
                 Use this symbol to represent the whitespace character.
";
source "$(pwd)/utils/parse_options.inc.sh" || exit 1;
#[ $# -ne 3 ] && echo "$help_message" >&2 && exit 1;

# We will use pypy, if available, since it is much faster.
tokenize_cmd="./utils/nltk_tokenize.py";
which pypy &> /dev/null && tokenize_cmd="pypy $tokenize_cmd";

mkdir -p data/lang/{char,word}/"$partition";

# Prepare word-level transcripts.
[ "$overwrite" = false -a -s "data/lang/word/$partition/all.txt" ] ||
awk '$1 !~ /^#/' "data/original/$partition.txt" | cut -d\  -f1,9- |
awk '{ $1=$1"|"; print; }' |
# Some words include spaces (e.g. "B B C" -> "BBC"), remove them.
sed -r 's| +||g' |
# Replace character | with whitespaces.
tr \| \  |
# Some contractions where separated from the words to reduce the vocabulary
# size. These separations are unreal, we join them (e.g. "We 'll" -> "We'll").
sed 's/ '\''\(s\|d\|ll\|m\|ve\|t\|re\|S\|D\|LL\|M\|VE\|T\|RE\)\b/'\''\1/g' |
sort -k1 > "data/lang/word/$partition/all.txt" ||
( echo "ERROR: Creating file data/lang/word/$partition/all.txt" >&2 && exit 1 );

# Prepare character-level transcripts.
[ "$overwrite" = false -a -s "data/lang/char/$partition/all.txt" ] ||
awk -v ws="$wspace" '{
  printf("%s", $1);
  for(i=2;i<=NF;++i) {
    for(j=1;j<=length($i);++j) {
      printf(" %s", substr($i, j, 1));
    }
    if (i < NF) printf(" %s", ws);
  }
  printf("\n");
}' "data/lang/word/$partition/all.txt" |
sort -k1 > "data/lang/char/$partition/all.txt" ||
( echo "ERROR: Creating file data/lang/char/$partition/all.txt" >&2 && exit 1 );

# Add whitespace boundaries to the character-level transcripts.
# Note: Actually not used, so far.
[ "$overwrite" = false -a -s "data/lang/char/$partition/all_wspace.txt" ] ||
awk -v ws="$wspace" '{ $1=$1" "ws; printf("%s %s\n", $0, ws); }' \
  "data/lang/char/$partition/all.txt" \
  > "data/lang/char/$partition/all_wspace.txt" ||
( echo "ERROR: Creating file data/lang/char/$partition/all_wspace.txt" >&2 &&
  exit 1 );

# Extract characters list for training.
mkdir -p "train/$partition";
[ "$overwrite" = false -a -s train/$partition/syms.txt ] ||
cut -d\  -f2- data/lang/char/$partition/all.txt | tr \  \\n | sort | uniq |
awk -v ws="$wspace" 'BEGIN{
  printf("%-12s %d\n", "<eps>", 0);
  printf("%-12s %d\n", "<ctc>", 1);
  printf("%-12s %d\n", ws, 2);
  N = 3;
}$1 != ws{
  printf("%-12s %d\n", $1, N++);
}' > train/$partition/syms.txt ||
( echo "ERROR: Creating file train/$partition/syms.txt" >&2 && exit 1 );

# Split files into different partitions (train, test, valid).
mkdir -p data/lang/{char,word}/"$partition/aachen";
for p in "$partition/aachen"/{tr,te,va}; do
  join -1 1 "data/part/$p.lst" "data/lang/char/$partition/all.txt" \
    > "data/lang/char/$p.txt" ||
  ( echo "ERROR: Creating file data/lang/char/$p.txt" >&2 && exit 1 );
  join -1 1 "data/part/$p.lst" "data/lang/word/$partition/all.txt" \
    > "data/lang/word/$p.txt" ||
  ( echo "ERROR: Creating file data/lang/word/$p.txt" >&2 && exit 1 );
  join -1 1 "data/part/$p.lst" "data/lang/char/$partition/all_wspace.txt" \
    > "data/lang/char/${p}_wspace.txt" ||
  ( echo "ERROR: Creating file data/lang/char/${p}_wspace.txt" >&2 && exit 1 );
done;




exit 0;

# Check required files
for f in  "$tdir/iam/"{lines,sentences,forms}.txt \
  "$tdir/charmap"  "$ldir/te.lst" "$ldir/tr.lst" "$ldir/va.lst"; do
  [ ! -f "$f" ] && echo "File \"$f\" not found!" >&2 && exit 1;
done;



# Create output directories.
mkdir -p "$odir"/{char,word}/iam/{lines,sentences,forms};

################################################################################
## PREPARE WORD-LEVEL FILES
################################################################################

# NOTE: WE DO NOT need to put abbrev. like 's, 't, 'd, etc. together with their
# word, since the original IAM transcripts where modified to according to this.
#
# Tokenize using NLTK-based tokenizer (Treebank tokenizer) and write the
# information about the word boundaries. This information is useful in order to
# create the lexicon. There are tokens that should not be preceded, in some
# cases, by a whitespace in the lexicon. Output files of this step are:
# $odir/${c}_tokenized.txt and $odir/${c}_boundaries.txt.

tmpf="$(mktemp)";
for p in te tr va; do
  for c in forms lines sentences; do
    # Get lines from the current partition: Basically, we filter by form ID
    # since forms are mutually exclusive from the different partitions.
    [[ "$overwrite" = false &&
        -s "$odir/word/iam/$c/$p.txt" &&
        -s "$odir/word/iam/$c/${p}_tokenized.txt" &&
        -s "$odir/word/iam/$c/${p}_boundaries.txt" &&
        ( ! "$odir/word/iam/$c/$p.txt" -ot "$tdir/iam/$c.txt" ) &&
        ( ! "$odir/word/iam/$c/${p}_tokenized.txt" -ot "$tdir/iam/$c.txt" ) &&
        ( ! "$odir/word/iam/$c/${p}_boundaries.txt" -ot "$tdir/iam/$c.txt" ) ]] ||
    awk -v PF="$ldir/$p.lst" '
function getFormID(s) {
  return gensub(/^([a-z0-9]+-[a-z0-9]+)-.+$/, "\\1", "g", s);
}
BEGIN{ while((getline < PF) > 0) { P[getFormID($1)]=1; } }
(getFormID($1) in P){ print; }' "$tdir/iam/$c.txt" > "$odir/word/iam/$c/$p.txt";
    # Tokenize partition-specific files.
    [[ "$overwrite" = false &&
        -s "$odir/word/iam/$c/${p}_tokenized.txt" &&
        -s "$odir/word/iam/$c/${p}_boundaries.txt" &&
        ( ! "$odir/word/iam/$c/${p}_tokenized.txt" -ot "$tdir/word/iam/$c/$p.txt" ) &&
        ( ! "$odir/word/iam/$c/${p}_boundaries.txt" -ot "$tdir/word/iam/$c/$p.txt" ) ]] ||
    awk '{$1=""; print; }' "$odir/word/iam/$c/$p.txt" |
    ./utils/nltk_tokenize.py \
      --write-boundaries "$odir/word/iam/$c/${p}_boundaries.txt" > "$tmpf";
    cut -d\  -f1 "$odir/word/iam/$c/$p.txt" | paste - "$tmpf" > \
      "$odir/word/iam/$c/${p}_tokenized.txt" ||
    ( echo "An error ocurred processing file \"$f\"!" >&2 && exit 1; );
  done;
done;



################################################################################
## PREPARE CHARACTER-LEVEL FILES
################################################################################

for p in te tr va; do
  for c in forms lines sentences; do
    # Get lines from the current partition: Basically, we filter by form ID
    # since forms are mutually exclusive from the different partitions.
    # And split characters into separate tokens.
    [[ "$overwrite" = false &&
        -s "$odir/char/iam/$c/$p.txt" &&
        ( ! "$odir/char/iam/$c/$p.txt" -ot "$tdir/iam/$c.txt" ) ]] ||
    awk -v PF="$ldir/$p.lst" -v ws="$wspace" '
function getFormID(s) {
  return gensub(/^([a-z0-9]+-[a-z0-9]+)-.+$/, "\\1", "g", s);
}
BEGIN{ while((getline < PF) > 0) { P[getFormID($1)]=1; } }
(getFormID($1) in P){
  printf("%s", $1);
  for (i=2;i<=NF;++i) {
    for (j=1;j<=length($i);++j) {
      printf(" %s", substr($i, j, 1));
    }
    if (i < NF) printf(" %s", ws);
  }
  printf("\n");
}' "$tdir/iam/$c.txt" > "$odir/char/iam/$c/$p.txt";
    # Replace characters with their mapping.
    [[ "$overwrite" = false &&
        -s "$odir/char/iam/$c/${p}_normalized.txt" &&
        ( ! "$odir/char/iam/$c/${p}_normalized.txt" -ot "$tdir/char/iam/$c/$p.txt" ) ]] ||
    awk -v CMF="$tdir/charmap" -v AW="$add_border_wspace" -v ws="$wspace" '
BEGIN {
  while((getline < CMF) > 0) { c=$1; $1=""; gsub(/^\ +/, ""); M[c]=$0; }
}{
  if (AW == "true") $1=$1" "ws;
  for (i=2; i<=NF; ++i) { if ($i in M) $i=M[$i]; }
  if (AW == "true") $NF=$NF" "ws;
  print;
}' "$odir/char/iam/$c/$p.txt" |
    sed -r 's|\ +| |g' > "$odir/char/iam/$c/${p}_normalized.txt";
  done;
done;

exit 0;
