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
height=120;
num_procs=$(nproc);
help_message="
Usage: ${0##*/} [options]

Options:
  --height     : (type = integer, default = $height)
                 Scale lines to have this height, keeping the aspect ratio
                 of the original image.
  --num_procs  : (type = integer, default = $num_procs)
  --overwrite  : (type = boolean, default = $overwrite)
                 Overwrite previously created files.
";
source "$(pwd)/utils/parse_options.inc.sh" || exit 1;

mkdir -p data/lang/{char,word};

# Put all word-level transcripts into a single file.
[ -s data/lang/word/all.txt ] ||
for f in data/original/contestHTRtS{/BenthamData,-test}/Transcriptions/*.txt; do
  bn="$(basename "$f" .txt)";
  echo -n "$bn " && head -n1 "$f" && echo "";
done | gawk 'NF > 0' |
sed -r 's|<gap/>|~|g;s|\ +| |g;s|^ ||g;s| $||g' |
sort -k1 > data/lang/word/all.txt ||
( echo "ERROR: Creating file data/lang/word/all.txt" >&2 && exit 1 );

# Put all char-level transcripts into a single file.
[ -s data/lang/char/all.txt ] ||
cut -d\  -f2- data/lang/word/all.txt |
tr \  \@ | sed -r 's|(.)|\1 |g;s|\ +$||g' |
paste -d\  <(cut -d\  -f1 data/lang/word/all.txt) - |
sort -k1 > data/lang/char/all.txt ||
( echo "ERROR: Creating file data/lang/char/all.txt" >&2 && exit 1 );

# Split transcript files into training, validation and test partitions.
orig_prefix=data/original/contestHTRtS;
for lvl in char word; do
  [ -s data/lang/$lvl/tr.txt ] ||
  join <(sort $orig_prefix/BenthamData/Partitions/TrainLines.lst) \
    data/lang/$lvl/all.txt > data/lang/$lvl/tr.txt ||
  ( echo "ERROR: Creating file data/lang/$lvl/tr.txt" >&2 && exit 1 );
  [ -s data/lang/$lvl/va.txt ] ||
  join <(sort $orig_prefix/BenthamData/Partitions/ValidationLines.lst) \
    data/lang/$lvl/all.txt > data/lang/$lvl/va.txt ||
  ( echo "ERROR: Creating file data/lang/$lvl/va.txt" >&2 && exit 1 );
  [ -s data/lang/$lvl/te.txt ] ||
  join <(sort $orig_prefix-test/Partitions/TestLines.lst) \
    data/lang/$lvl/all.txt > data/lang/$lvl/te.txt ||
  ( echo "ERROR: Creating file data/lang/$lvl/te.txt" >&2 && exit 1 );
done;

# Process images: enhance contrast, crop, and resize to fixed height
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

function process_job () {
  local bn="$(basename "$1" .png)";
  [[ "$overwrite" = false && -s "data/imgs/lines/$bn.jpg" ]] && return 0;
  imgtxtenh -d 118.110 "$1" png:- |
  convert png:- -deskew 40% \
    -bordercolor white -border 5 -trim \
    -bordercolor white -border 20x0 +repage \
    -strip "data/imgs/lines/$bn.jpg" ||
  { echo "ERROR: Processing image $1" >&2 && return 1; }
  # Resize image
  convert "data/imgs/lines/$bn.jpg" -resize "x$height" +repage \
    -strip "data/imgs/lines_h$height/$bn.jpg" ||
  { echo "ERROR: Processing image $1" >&2 && return 1; }
  return 0;
}

mkdir -p data/imgs/lines "data/imgs/lines_h$height";
[[ "$overwrite" = false &&
   -s data/lists/te.lst  && -s data/lists/tr.lst  && -s data/lists/va.lst ]] ||
(
  tmpd="$(mktemp -d)";
  bkg_pids=();
  for f in $orig_prefix/BenthamData/Images/Lines/*.png \
    $orig_prefix-test/Images/Lines-test/*.png; do
    process_job "$f" &> "$tmpd/${#bkg_pids[@]}" &
    bkg_pids+=("$!");
    # Wait for jobs to finish when number of running jobs = number of processors
    if [ "${#bkg_pids[@]}" -eq "$num_procs" ]; then
      wait_jobs "${bkg_pids[@]}" || exit 1;
      bkg_pids=();
    fi;
  done;
  wait_jobs "${bkg_pids[@]}" || exit 1;
  rm -r "$tmpd";
)

# Create lists of images
mkdir -p data/lists;
for p in te tr va; do
  [ -s "data/lists/$p.lst" ] ||
  gawk '{ print "data/imgs/lines/"$1".jpg" }' "data/lang/char/$p.txt" \
    > "data/lists/$p.lst" ||
  { echo "ERROR: Creating list of images data/lists/$p.lst" >&2 && exit 1; }
  [ -s "data/lists/${p}_h$height.lst" ] ||
  gawk -v height=$height '{ print "data/imgs/lines_h"height"/"$1".jpg" }' \
    "data/lang/char/$p.txt" > "data/lists/${p}_h$height.lst" ||
  { echo "ERROR: Creating list of images data/lists/${p}_h$height.lst" >&2 &&
    exit 1; }
done;

# Create list of characters for training
mkdir -p train;
[ -s train/syms.txt ] ||
cut -d\  -f2- data/lang/char/{tr,va}.txt | tr \  \\n | sort | uniq |
gawk 'BEGIN{
  printf("%-12s %d\n", "<eps>", 0);
  printf("%-12s %d\n", "<ctc>", 1);
  N = 2;
}{
  printf("%-12s %d\n", $1, N++);
}' > train/syms.txt ||
( echo "ERROR: Creating file train/syms.txt" >&2 && exit 1 );

exit 0;
