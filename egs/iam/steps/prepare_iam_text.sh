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
wspace="<space>";
help_message="
Usage: ${0##*/} [options]

Options:
  --overwrite  : (type = boolean, default = $overwrite)
                 Overwrite previously created files.
  --wspace     : (type = string, default \"$wspace\")
                 Use this symbol to represent the whitespace character.
";
source "$(pwd)/utils/parse_options.inc.sh" || exit 1;

# We will use pypy, if available, since it is much faster.
tokenize_cmd="./utils/nltk_tokenize.py";
if which pypy &> /dev/null; then tokenize_cmd="pypy $tokenize_cmd"; fi;

mkdir -p data/lang/{char,word}/{lines,forms};

# Prepare word-level transcripts.
[ "$overwrite" = false -a -s "data/lang/word/lines/all.txt" ] ||
awk '$1 !~ /^#/' "data/original/lines.txt" | cut -d\  -f1,9- |
awk '{ $1=$1"|"; print; }' |
# Some words include spaces (e.g. "B B C" -> "BBC"), remove them.
sed -r 's| +||g' |
# Replace character | with whitespaces.
tr \| \  |
# Some contractions where separated from the words to reduce the vocabulary
# size. These separations are unreal, we join them (e.g. "We 'll" -> "We'll").
sed 's/ '\''\(s\|d\|ll\|m\|ve\|t\|re\|S\|D\|LL\|M\|VE\|T\|RE\)\b/'\''\1/g' |
sort -k1 > "data/lang/word/lines/all.txt" ||
{ echo "ERROR: Creating file data/lang/word/lines/all.txt" >&2 && exit 1; }

# Prepare character-level transcripts.
[ "$overwrite" = false -a -s "data/lang/char/lines/all.txt" ] ||
awk -v ws="$wspace" '{
  printf("%s", $1);
  for(i=2;i<=NF;++i) {
    for(j=1;j<=length($i);++j) {
      printf(" %s", substr($i, j, 1));
    }
    if (i < NF) printf(" %s", ws);
  }
  printf("\n");
}' "data/lang/word/lines/all.txt" |
sort -k1 > "data/lang/char/lines/all.txt" ||
{ echo "ERROR: Creating file data/lang/char/lines/all.txt" >&2 && exit 1; }

# Add whitespace boundaries to the character-level transcripts.
# Note: Actually not used, but could be used in the future.
[ "$overwrite" = false -a -s "data/lang/char/lines/all_wspace.txt" ] ||
awk -v ws="$wspace" '{ $1=$1" "ws; printf("%s %s\n", $0, ws); }' \
  "data/lang/char/lines/all.txt" \
  > "data/lang/char/lines/all_wspace.txt" ||
{ echo "ERROR: Creating file data/lang/char/lines/all_wspace.txt" >&2 &&
  exit 1; }

# Extract characters list for training.
mkdir -p "train/lines";
[ "$overwrite" = false -a -s "train/lines/syms.txt" ] ||
cut -d\  -f2- "data/lang/char/lines/all.txt" | tr \  \\n | sort | uniq |
awk -v ws="$wspace" 'BEGIN{
  printf("%-12s %d\n", "<eps>", 0);
  printf("%-12s %d\n", "<ctc>", 1);
  printf("%-12s %d\n", ws, 2);
  N = 3;
}$1 != ws{
  printf("%-12s %d\n", $1, N++);
}' > "train/lines/syms.txt" ||
{ echo "ERROR: Creating file train/lines/syms.txt" >&2 && exit 1; }

# Split files into different partitions (train, test, valid).
mkdir -p data/lang/{char,word}/lines/aachen;
for p in lines/aachen/{tr,te,va}; do
  join -1 1 "data/part/$p.lst" "data/lang/char/lines/all.txt" \
    > "data/lang/char/$p.txt" ||
  { echo "ERROR: Creating file data/lang/char/$p.txt" >&2 && exit 1; }
  join -1 1 "data/part/$p.lst" "data/lang/word/lines/all.txt" \
    > "data/lang/word/$p.txt" ||
  { echo "ERROR: Creating file data/lang/word/$p.txt" >&2 && exit 1; }
  join -1 1 "data/part/$p.lst" "data/lang/char/lines/all_wspace.txt" \
    > "data/lang/char/${p}_wspace.txt" ||
  { echo "ERROR: Creating file data/lang/char/${p}_wspace.txt" >&2 && exit 1; }
done;

# Tokenize using NLTK-based tokenizer (Treebank tokenizer) and write the
# information about the word boundaries. This information is useful in order to
# create the lexicon. There are tokens that should not be preceded, in some
# cases, by a whitespace in the lexicon. Output files of this step are:
# $odir/${c}_tokenized.txt and $odir/${c}_boundaries.txt.
for p in lines/aachen/{tr,te,va}; do
  tok="data/lang/word/${p}_tokenized.txt";
  bnd="data/lang/word/${p}_boundaries.txt";
  [[ "$overwrite" = false && -s "$tok" && -s "$bnd" &&
    ( ! "data/lang/word/$p.txt" -nt "$tok" ) &&
    ( ! "data/lang/word/$p.txt" -nt "$bnd" ) ]] ||
  cut -d\  -f2- "data/lang/word/$p.txt" |
  $tokenize_cmd --write-boundaries "$bnd" |
  paste -d\  <(cut -d\  -f1 "data/lang/word/$p.txt") - > "$tok" ||
  { echo "ERROR: Creating file $tok" >&2 && exit 1; }
done;

mkdir -p data/lang/{char,word}/forms/aachen;
for p in tr te va; do
  txtw="data/lang/word/forms/aachen/$p.txt";
  txtc="data/lang/char/forms/aachen/$p.txt";
  tok="data/lang/word/forms/aachen/${p}_tokenized.txt";
  bnd="data/lang/word/forms/aachen/${p}_boundaries.txt";
  # Get the word-level transcript of the whole form.
  [[ "$overwrite" = false && -s "$txtw" &&
      ( ! "$txtw" -ot "data/lang/word/lines/aachen/$p.txt" ) ]] ||
  awk 'BEGIN{ sent_id=""; }{
    if (match($0, /^([^ ]+)-[0-9]+ (.+)$/, A)) {
      if (A[1] != sent_id) {
        if (sent_id != "") printf("\n");
        printf("%s %s", A[1], A[2]);
        sent_id = A[1];
      } else {
        printf(" %s", A[2]);
      }
    }
  }END{ if (sent_id != "") printf("\n"); }' \
    "data/lang/word/lines/aachen/$p.txt" > "$txtw" ||
  { echo "ERROR: Creating file \"$txtw\"!" >&2 && exit 1; }
  # Tokenize the joint sentences.
  [[ "$overwrite" = false && -s "$tok" && -s "$bnd" &&
      ( ! "$txtw" -nt "$tok" ) && ( ! "$txtw" -nt "$bnd" ) ]] ||
  cut -d\  -f2- "$txtw" |
  $tokenize_cmd --write-boundaries "$bnd" |
  paste -d\  <(cut -d\  -f1 "$txtw") - > "$tok" ||
  { echo "ERROR: Creating file $tok" >&2 && exit 1; }
  # Prepare character-level transcripts.
  [ "$overwrite" = false -a -s "$txtc" ] ||
  awk -v ws="$wspace" '{
    printf("%s", $1);
    for(i=2;i<=NF;++i) {
      for(j=1;j<=length($i);++j) {
        printf(" %s", substr($i, j, 1));
      }
      if (i < NF) printf(" %s", ws);
    }
    printf("\n");
  }' "$txtw" > "$txtc" ||
  { echo "ERROR: Creating file $txtc" >&2 && exit 1; }
done;

exit 0;
