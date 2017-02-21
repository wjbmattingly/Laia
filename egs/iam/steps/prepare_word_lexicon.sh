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

wspace="<space>"
help_message="
Usage: ${0##*/} charmap boundaries1 [boundaries2 ...]

Description:
  Use this file to create the lexicon mapping words into seqs characters.
  This script takes as input a character map and a list of boundary files.
  The lexicon adds (or not) a whitespace character at the beginning of each
  word, according to the boundary file.

Arguments:
  charmap         : File containing the mapping from characters to HMMs
                    (e.g. \"data/text/charmap\").
  boundaries1 ... : File(s) containing the boundaries of each word in the text
                    data (e.g. \"exper/htr/lang/word/brown_boundaries.txt\").

Options:
  --wspace        : (type = string, default = \"$wspace\")
                    Use this symbol to represent the whitespace character.
";
source "$(pwd)/utils/parse_options.inc.sh" || exit 1;
[ $# -lt 2 ] && echo "$help_message" >&2 && exit 1;

charmap="$1"; shift;
boundaries=();
while [ $# -gt 0 ]; do boundaries+=("$1"); shift; done;

cat "${boundaries[@]}" |
python -c "
import sys

L = {}
for line in sys.stdin:
  line = line.split()
  if len(line) < 1: continue
  w = line[0]
  c = float(line[1] if len(line) > 1 else 1.0)
  p = line[2:] if len(line) > 2 else w
  # Remove final whitespace, since we are going to model only whitespace
  # at the beginning of the word.
  if p[-1] == '\\s': p = p[:-1]
  p = tuple(p)
  # Update lexicon counter
  if w not in L: L[w] = {}
  if p not in L[w]: L[w][p] = c
  else: L[w][p] += c


for w in L:
  Zw = sum(L[w].itervalues())
  for (p, c) in L[w].iteritems():
    print '%-25s    %f' % (w, c / Zw),
    if p[0] == '\\s':
      print '$wspace',
      p = p[1:]
    for c in unicode(p[0], 'utf-8'):
      print c,
    print ''
" | awk -v CMF="$charmap" '
BEGIN{
  while((getline < CMF) > 0) { c = $1; $1=""; gsub(/^\ +/, ""); M[c] = $0; }
}{
  printf("%-25s    %f", $1, $2);
  for (i=3; i<=NF; ++i) { if ($i in M) $i=M[$i]; printf(" %s", $i); }
  printf("\n");
}' | sort -V;
exit 0;