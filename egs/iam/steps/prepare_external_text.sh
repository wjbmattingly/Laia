#!/bin/bash
set -e;

# Directory where the prepare.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/steps" != "$SDIR" ] && \
    echo "Please, run this script from the experiment top directory!" >&2 && \
    exit 1;
[ ! -f "$(pwd)/utils/parse_options.inc.sh" ] && \
    echo "Missing $(pwd)/utils/parse_options.inc.sh file!" >&2 && exit 1;

add_border_wspace=false;
overwrite=false;
wspace="<space>";
help_message="
Usage: ${0##*/} [options] text1 [text2 ...] charmap output_dir

Arguments:
  text1 ...    : External text files to process
                 (e.g. data/text/brown.txt data/text/wellington.txt).
  output_dir   : Output directory containing all the processed files
                 (e.g. exp/htr/lang).

Options:
  --add_border_wspace : (type = boolean, default = $add_border_wspace)
                        If true, add a whitespace symbol at the start and end of
                        each line, in the character-level normalized text.
  --overwrite         : (type = boolean, default = $overwrite)
                        Overwrite previously created files.
  --wspace            : (type = string, default \"$wspace\")
                        Use this symbol to represent the whitespace character.
";
source "$(pwd)/utils/parse_options.inc.sh" || exit 1;
[ $# -lt 3 ] && echo "$help_message" >&2 && exit 1;

# Parse arguments
corpora=();
while [ $# -gt 2 ]; do
  [[ ( ! -s "$1" ) ]] && echo "File \"$1\" does not exist!" && exit 1;
  corpora+=("$1");
  shift;
done;
charmap="$1"
odir="$2";
mkdir -p "$odir"/{word,char};

# We will use pypy, if available, since it is much faster.
tokenize_cmd="./utils/nltk_tokenize.py";
which pypy &> /dev/null && tokenize_cmd="pypy $tokenize_cmd";


################################################################################
## PREPARE WORD-LEVEL FILES
################################################################################

# 1. We convert some special sequences to UTF-8 characters, such as a*?1 -> ä,
# a*?2 -> á, a*?3 -> à, n*?4 -> ñ, etc.
# 2. Since IAM does not contain these characters, trasliterate all UTF-8 codes
# to reduce the number of tokens in the LM.
# 3. Put abbrev. like 's, 't, 'd, etc. together with their word, since NLTK will
# tokenize this later. (Save this as $odir/word/$c.txt).
# 4. Finally, tokenize using NLTK default tokenizer (Treebank tokenizer) and
# write the information about the word boundaries. This information is useful
# in order to create the lexicon. There are tokens that should not be preceded,
# in some cases, by a whitespace in the lexicon. Output files of this step are:
# $odir/word/${c}_tokenized.txt and $odir/word/${c}_boundaries.txt.

for f in "${corpora[@]}"; do
  c="$(basename "$f" .txt)";
  [[ "$overwrite" = false &&
      -s "$odir/word/$c.txt" &&
      -s "$odir/word/${c}_tokenized.txt" &&
      -s "$odir/word/${c}_boundaries.txt" &&
      ( ! "$odir/word/$c.txt" -ot "$f" ) &&
      ( ! "$odir/word/${c}_tokenized.txt" -ot "$f" ) &&
      ( ! "$odir/word/${c}_boundaries.txt" -ot "$f" ) ]] ||
  cat "$f" |
  ./steps/make_utf8_WLOB.sh |
  iconv -f utf-8 -t ascii//TRANSLIT |
  sed 's/ '\''\(s\|d\|ll\|m\|ve\|t\|re\|S\|D\|LL\|M\|VE\|T\|RE\)\b/'\''\1/g' |
  tee "$odir/word/$c.txt" |
  ./utils/nltk_tokenize.py \
    --write-boundaries "$odir/word/${c}_boundaries.txt" \
    > "$odir/word/${c}_tokenized.txt" ||
  ( echo "An error ocurred processing file \"$f\"!" >&2 && exit 1; );
done;



################################################################################
## PREPARE CHARACTER-LEVEL FILES
################################################################################

for f in "${corpora[@]}"; do
  c="$(basename "$f" .txt)";
  [[ "$overwrite" = false &&
      -s "$odir/char/$c.txt" &&
      -s "$odir/char/${c}_tokenized.txt" &&
      -s "$odir/char/${c}_boundaries.txt" &&
      ( ! "$odir/char/$c.txt" -ot "$odir/word/$c.txt" ) &&
      ( ! "$odir/char/${c}_tokenized.txt" -ot "$odir/word/$c.txt" ) &&
      ( ! "$odir/char/${c}_boundaries.txt" -ot "$odir/word/$c.txt" ) ]] ||
  awk -v ws="$wspace" '
{
  for (i=1;i<=NF;++i) {
    for (j=1;j<=length($i);++j) {
      printf(" %s", substr($i, j, 1));
    }
    if (i < NF) printf(" %s", ws);
  }
  printf("\n");
}' "$odir/word/$c.txt" | sed -r 's|\ +| |g' | tee "$odir/char/$c.txt" |
  awk -v CMF="$charmap" -v AW="$add_border_wspace" -v ws="$wspace" '
BEGIN {
  while((getline < CMF) > 0) { c=$1; $1=""; gsub(/^\ +/, ""); M[c]=$0; }
}{
  for (i=1; i<=NF; ++i) { if ($i in CM) $i=M[$i]; }
  if (AW == "true") $1=ws" "$1;
  if (AW == "true") $NF=$NF" "ws;
  print;
}' | sed -r 's|\ +| |g' > "$odir/char/${c}_normalized.txt";
done;

exit 0;
