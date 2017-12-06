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
Usage: ${0##*/} [options] <htk-dict> <input-phone-table> <input-word-table>
       <output-word-table> <output-lexicon-int>
";
source utils/parse_options.inc.sh || exit 1;
[ $# -ne 5 ] && echo "$help_message" >&2 && exit 1;

for f in "$1" "$2" "$3"; do
  [ ! -f "$f" ] && echo "ERROR: File \"$f\" does not exist!" >&2 && exit 1;
done;

gawk -v PIT="$2" -v WIT="$3" -v WOT="$4" 'BEGIN{
  while ((getline < PIT) > 0) {
    phon2int[$1] = $2;
  }
  while ((getline < WIT) > 0) {
    word2int[$1] = $2;
  }
  N = 0;
}{
  # Get input lexicon word
  if (substr($1, 1, 1) == "\"") {
    w_inp = substr($1, 2, length($1) - 2); gsub(/\\/, "", w_inp);
  } else {
    w_inp = $1; gsub(/\\/, "", w_inp);
  }
  # Get output lexicon word (a.k.a. pronunciation) and its probability
  if (substr($2, 1, 1) == "[") {
    w_out = substr($2, 2, length($2) - 2);
    p=$3;
  } else {
    w_out = w_inp;
    p=$2;
  }

  if (!(w_inp in word2int)) {
    print "Word \""w_inp"\" was not found in the input word table!" > "/dev/stderr";
    next;
  } else {
    w_inp = word2int[w_inp];
  }

  if (!(w_out in pron2int)) {
    pron2int[w_out] = ++N
  }
  w_out = pron2int[w_out];

  pron = "";
  for (i = 4; i <= NF; ++i) {
    if (!($i in phon2int)) {
      print "Phone \""$i"\" was not found in the input phone table!" > "/dev/stderr";
      exit(1);
    } else {
      pron = sprintf("%s %d", pron, phon2int[$i]);
    }
  }

  key = sprintf("%d %s", w_inp, pron);
  if (p > MAX_P[key]) {
    MAX_P[key] = p;
    MAX_E[key] = sprintf("%d %d%s", w_inp, w_out, pron);
  }
}END{
  # Print lexicon for lattice-align-word-lexicon
  cmd_lex = "sort -n";
  for (k in MAX_E) {
    print MAX_E[k] | cmd_lex
  }

  # Print output words table
  cmd_tab = "sort -nk2 > "WOT;
  printf("%25-s %d\n", "<eps>", 0) | cmd_tab;
  for (w in pron2int) {
    printf("%25-s %d\n", w, pron2int[w]) | cmd_tab;
  }
}' "$1" > "$5";

exit 0;
