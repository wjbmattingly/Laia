#!/bin/bash
set -e;
export LC_NUMERIC=C;

# Directory where the prepare.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/steps" != "$SDIR" ] && \
    echo "Please, run this script from the experiment top directory!" >&2 && \
    exit 1;
[ ! -f "$(pwd)/utils/parse_options.inc.sh" ] && \
    echo "Missing $(pwd)/utils/parse_options.inc.sh file!" >&2 && exit 1;

overwrite=false;
help_message="
Usage: ${0##*/} [options]
";
source utils/parse_options.inc.sh || exit 1;
[ $# -ne 0 ] && echo "$help_message" >&2 && exit 1;

[ ! -f "data/original/Bentham-Batch1_newTok.dic" ] &&
echo "ERROR: File \"$f\" was not found!" >&2 && exit 1;

awk 'BEGIN{
  printf("%20-s %f    @\n", "</s>", 1.0);
}{
  # Get lexicon word
  w = substr($1, 2, length($1) - 2); gsub(/\\/, "", w);
  # Lexicon probability
  p=$3;

  if (w == "<s>") next;
  if (w == "</s>") next;

  printf("%20-s %f   ", w, p);
  for (i = 4; i <= NF; ++i) {
    if ($i == "<is>") {
      printf(" @");
    } else if ($i == "<fs>") {
      // omit this symbol
    } else if ($i == "<GAP>") {
      printf(" ~");
    } else if ($i == "<quote>") {
      printf(" '\''");
    } else if ($i == "<dquote>") {
      printf(" \"");
    } else if ($i == "^") {
      // omit this symbol
    } else {
      printf(" %s", $i);
    }
  }
  printf("\n");
}' data/original/Bentham-Batch1_newTok.dic \
  > data/lang/word/original_lexiconp.txt;
exit 0;
