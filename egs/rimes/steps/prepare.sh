#!/bin/bash
set -e;
export LC_NUMERIC=C;

# Directory where this script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/steps" != "$SDIR" ] &&
  echo "Please, run this script from the experiment top directory!" >&2 &&
  exit 1;
[ ! -f "$(pwd)/utils/parse_options.inc.sh" ] &&
  echo "Missing $(pwd)/utils/parse_options.inc.sh file!" >&2 && exit 1;

height=128;
nproc="$(nproc)";
overwrite=false;
help_message="
Usage: ${0##*/} [options]

Options:
  --height     : (type = integer, default = $height)
                 Scale lines to have this height, keeping the aspect ratio
                 of the original image.
  --nproc      : (type = integer, default = $nproc)
                 Use this number of concurrent processes to prepare the images.
  --overwrite  : (type = boolean, default = $overwrite)
                 Overwrite previously created files.
";
source "$(pwd)/utils/parse_options.inc.sh" || exit 1;

# Check that all required files exist!
for f in data/a2ia/training_2011.patched.xml data/a2ia/eval_2011_annotated.xml; do
  [ ! -s "$f" ] && echo "ERROR: File \"$f\" does not exist!" >&2 && exit 1;
done;

# Check ImageMagick
which convert &> /dev/null ||
{ echo "ERROR: ImageMagick's convert is required, please install it!" >&2 &&
  exit 1; }

# Check Mauricio's imgtxtenh
which imgtxtenh &> /dev/null ||
{ echo "ERROR: imgtxtenh tool is required, please install it from https://github.com/mauvilsa/imgtxtenh" >&2 && exit 1; }

# Extract info from the XML files.
mapfile -t xmldata < <(
  for f in data/a2ia/training_2011.patched.xml data/a2ia/eval_2011_annotated.xml; do
    python -c "
import re
import sys
from xml.sax import parse
from xml.sax.handler import ContentHandler
from os.path import basename

class RIMESHandler(ContentHandler):
  def startElement(self, name, attrs):
    if name == 'SinglePage':
      self.filename = attrs['FileName']
      self.linenum  = 0
    elif name == 'Line':
      print basename(self.filename.encode('utf-8')), self.linenum, \
        attrs['Left'], attrs['Top'], attrs['Right'], attrs['Bottom'], \
        attrs['Value'].encode('utf-8')
      self.linenum += 1

h = RIMESHandler()
parse(sys.stdin, h)
" < "$f";
  done;
);

# Create folders.
mkdir -p data/imgs/lines{,"_h$height"} data/lang/{lines,forms}/{char,word} \
  data/lists data/part/{lines,forms};
mkdir -p train;
tmpd="$(mktemp -d)";

function process_line () {
  bn="$1"; bn="${bn/.png/}";
  n="$(printf "%02d" $2)";  l="$3"; t="$4"; r="$5"; b="$6";
  w=$[r - l + 1];
  h=$[b - t + 1];
  i="data/a2ia/images_gray/$bn.png";
  o="data/imgs/lines/$bn-$n.jpg";
  # Extract image lines.
  [[ "$overwrite" = false && -s "$o" ]] ||
  convert "$i" -crop "${w}x${h}+${l}+${t}" +repage png:- |
  imgtxtenh -d 118.110 - png:- |
  convert png:- -colorspace Gray -deskew 40% \
    -bordercolor white -border 5 -trim \
    -bordercolor white -border 20x0 +repage \
    -strip "$o";
  # Resize image to have an height of $height pixels, keeping aspect ratio.
  oh="data/imgs/lines_h$height/$bn-$n.jpg";
  [[ "$overwrite" = false && -s "$oh" ]] ||
  convert "$o" -resize "x$height" +repage -strip "$oh";
}

function wait_jobs () {
  local n=0;
  while [ $# -gt 0 ]; do
    if ! wait "$1"; then
      echo "Failed image processing:" >&2 && cat "$tmpd/$n" >&2 && return 1
    fi;
    shift 1; ((++n));
  done;
  return 0;
}

# Extract line images and line transcripts.
bkg_pids=();
[[ "$overwrite" = false && -s data/lang/lines/word/all.txt &&
    -s "data/imgs/lines/eval2011-0-0.jpg" &&
    -s "data/imgs/lines_h$height/eval2011-0-0.jpg" ]] ||
for d in "${xmldata[@]}"; do
  d=($d);
  bn="${d[0]}"; bn="${bn/.png/}"; n="$(printf "%02d" ${d[1]})";
  if [[ "$overwrite" = true || ! -s "data/imgs/lines/$bn-$n.jpg" ||
	! -s "data/imgs/lines_h$height/$bn-$n.jpg" ]]; then
    process_line "${d[@]}" &> "$tmpd/${#bkg_pids[@]}" &
    bkg_pids+=("$!");
    [ "${#bkg_pids[@]}" -lt "$nproc" ] ||
    { wait_jobs "${bkg_pids[@]}" && bkg_pids=(); } ||  exit 1;
  fi;
  echo "$bn-$n" "${d[@]:6}"
done |
sed -r 's|¤||g' |  # This symbol does not actually exist in the images
sort -V > data/lang/lines/word/all.txt ||
{ echo "ERROR: Processing images failed, see \"$log\"!" >&2 && exit 1; }
wait_jobs "${bkg_pids[@]}" || exit 1;

# Get character-level transcription from the lines
[[ "$overwrite" = false && -s data/lang/lines/char/all.txt ]] ||
awk '{
  printf("%s", $1);
  for (i=2; i<=NF; ++i) {
    for (j=1; j<=length($i); ++j) {
      printf(" %s", substr($i, j, 1));
    }
    if (i < NF) printf(" <space>");
  }
  printf("\n");
}' data/lang/lines/word/all.txt > data/lang/lines/char/all.txt ||
exit 1;

# Extract FORM IDs for validation, training and test.
# Validation ...
[ "$overwrite" = false -a -s "data/part/forms/va.lst" ] ||
awk '$1 ~ /train/{ n = match($1, /^([^ ]+)-[0-9]+$/, A); print A[1]; }' \
  data/lang/lines/word/all.txt | sort -V | uniq |
shuf --random-source=data/lang/lines/word/all.txt |
head -n150 | sort -V > "data/part/forms/va.lst" ||
{ echo "ERROR: Creating file \"data/part/forms/va.lst\"!" >&2 && exit 1; }
# Training ...
[ "$overwrite" = false -a -s "data/part/forms/tr.lst" ] ||
awk '$1 ~ /train/{ n = match($1, /^([^ ]+)-[0-9]+$/, A); print A[1]; }' \
  data/lang/lines/word/all.txt | sort -V | uniq |
shuf --random-source=data/lang/lines/word/all.txt |
tail -n+151 | sort -V > "data/part/forms/tr.lst" ||
{ echo "ERROR: Creating file \"data/part/forms/tr.lst\"!" >&2 && exit 1; }
# Test ...
[ "$overwrite" = false -a -s "data/part/forms/te.lst" ] ||
awk '$1 ~ /eval/{ n = match($1, /^([^ ]+)-[0-9]+$/, A); print A[1]; }' \
  data/lang/lines/word/all.txt | sort -V | uniq > "data/part/forms/te.lst" ||
{ echo "ERROR: Creating file \"data/part/forms/te.lst\"!" >&2 && exit 1; }

for p in tr va te; do
  # Extract LINE IDs for partition $p
  [ "$overwrite" = false -a -s "data/part/lines/$p.lst" ] ||
  awk -v KF="data/part/forms/$p.lst" '
  BEGIN{
    while((getline < KF) > 0) { KEEP[$1] = 1; }
  }{
    if (match($1, /^([^ ]+)-[0-9]+$/, A) && A[1] in KEEP) print $1;
  }' data/lang/lines/word/all.txt > "data/part/lines/$p.lst";
  for t in char word; do
    # Extract LINE {char, word}-level transcript for partion $p
    join -1 1 <(sort -k1b,1 "data/part/lines/$p.lst") \
              <(sort -k1b,1 "data/lang/lines/$t/all.txt") |
    sort -V > "data/lang/lines/$t/$p.txt" ||
    { echo "ERROR: Creating file data/lang/lines/$t/$p.txt" >&2 && exit 1; }
    # Extract FORM {char,word}-level transcript for partition $p
    [[ "$overwrite" = false && -s "data/lang/forms/$t/$p.txt" ]] ||
    awk -v t="$t" 'BEGIN{ sent_id=""; }{
      if (match($0, /^([^ ]+)-[0-9]+ (.+)$/, A)) {
        if (A[1] != sent_id) {
          if (sent_id != "") printf("\n");
          printf("%s %s", A[1], A[2]);
          sent_id = A[1];
        } else {
          if (t == "char") printf(" %s", "<space>");
          printf(" %s", A[2]);
        }
      }
    }END{ if (sent_id != "") printf("\n"); }' \
      "data/lang/lines/$t/$p.txt" > "data/lang/forms/$t/$p.txt" ||
    { echo "ERROR: Creating file \"data/lang/forms/$t/$p.txt\"!" >&2 &&
      exit 1; }
  done;
  # TOKENIZE FORM/LINE word-level transcript for partition $p
  for t in lines forms; do
    [[ "$overwrite" = false && -s "data/lang/$t/word/${p}_tokenized.txt" &&
	-s "data/lang/$t/word/${p}_boundaries.txt" ]] ||
    cut -d\  -f2- "data/lang/$t/word/$p.txt" |
    ./steps/rimes_tokenize.py \
      --write-boundaries="data/lang/$t/word/${p}_boundaries.txt" |
    paste -d\  <(cut -d\  -f1 "data/lang/$t/word/$p.txt") - \
      > "data/lang/$t/word/${p}_tokenized.txt" ||
    { echo "ERROR: Creating file data/lang/$t/word/${p}_tokenized.txt" >&2 &&
      exit 1; }
  done;
  # Create lists of line images.
  [[ "$overwrite" = false && -s "data/lists/$p.lst" ]] ||
  awk '{ print "data/imgs/lines/"$1".jpg" }' "data/part/lines/$p.lst" |
  sort -V > "data/lists/$p.lst" ||
  { echo "ERROR: Creating file data/lists/${p}.lst" >&2 && exit 1; }
  [[ "$overwrite" = false && -s "data/lists/${p}_h$height.lst" ]] ||
  awk -v H="$height" '{ print "data/imgs/lines_h"H"/"$1".jpg" }' \
    "data/part/lines/$p.lst" |
  sort -V > "data/lists/${p}_h$height.lst" ||
  { echo "ERROR: Creating file data/lists/${p}_h${height}.lst" >&2 && exit 1; }
done;

# Create list of symbols for Laia training.
[[ "$overwrite" = false && -s train/syms.txt &&
    ( ! train/syms.txt -ot data/lang/lines/char/tr.txt ) &&
    ( ! train/syms.txt -ot data/lang/lines/char/va.txt ) ]] ||
cut -d\  -f2-  data/lang/lines/char/{tr,va}.txt | tr \  \\n | sort -V | uniq |
awk '
BEGIN{
  N = 0;
  printf("%-12s %d\n", "<eps>", N++);
  printf("%-12s %d\n", "<ctc>", N++);
  printf("%-12s %d\n", "<space>", N++);
}{
  if ($1 == "<space>") next;
  printf("%-12s %d\n", $1, N++);
}' > train/syms.txt || exit 1;

exit 0;
