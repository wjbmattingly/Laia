#!/bin/bash
set -e;

# Directory where the prepare.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/steps" != "$SDIR" ] && \
    echo "Please, run this script from the experiment top directory!" >&2 && \
    exit 1;
[ ! -f "$(pwd)/utils/parse_options.inc.sh" ] && \
    echo "Missing $(pwd)/utils/parse_options.inc.sh file!" >&2 && exit 1;

height=112;
nproc="$(nproc)";
overwrite=false;
help_message="
Usage: ${0##*/} [options]

Options:
  --height     : (type = integer, default = $height)
                 Scale lines to have this height, keeping the aspect ratio of
                 the original image.
  --nproc      : (type = integer, default = $nproc)
                 Use this number of concurrent processes to prepare the images.
  --overwrite  : (type = boolean, default = $overwrite)
                 Overwrite previously created files.
";
source "$(pwd)/utils/parse_options.inc.sh" || exit 1;

which imgtxtenh &> /dev/null ||
( echo "ERROR: Install https://github.com/mauvilsa/imgtxtenh" >&2 && exit 1 );
which convert &> /dev/null ||
( echo "ERROR: Install ImageMagick's convert" >&2 && exit 1 );

tmpd="$(mktemp -d)";

function process_line () {
  local bn="$(basename "$1" .png)";
  # Process image
  [ -s "data/imgs/lines/$bn.jpg" ] ||
  imgtxtenh -u mm -d 118.110 "$1" png:- |
  convert png:- -set option:deskew:auto-crop true -deskew 40% \
    -bordercolor white -border 5 -trim \
    -bordercolor white -border 20x0 +repage \
    -strip "data/imgs/lines/$bn.jpg" ||
  ( echo "ERROR: Processing image $1" >&2 && return 1 );
  # Resize image
  [ -s "data/imgs/lines_h$height/$bn.jpg" ] ||
  convert "data/imgs/lines/$bn.jpg" -resize "x$height" +repage \
    -strip "data/imgs/lines_h$height/$bn.jpg" ||
  ( echo "ERROR: Processing image $1" >&2 && return 1 );
  return 0;
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

# Enhance images with Mauricio's tool, deskew the line, crop white borders
# and resize to the given height.
mkdir -p data/imgs/lines{,_h$height};
bkg_pids=();
for f in data/original/lines/*.png; do
  bn="$(basename "$f" .png)";
  [ "$overwrite" = false -a -s "data/imgs/lines/$bn.jpg" -a \
    -s "data/imgs/lines_h$height/$bn.jpg" ] && continue;
  process_line "$f" &> "$tmpd/${#bkg_pids[@]}" &
  bkg_pids+=("$!");
  [ "${#bkg_pids[@]}" -lt "$nproc" ] ||
  { wait_jobs "${bkg_pids[@]}" && bkg_pids=(); } ||
  exit 1;
done;
wait_jobs "${bkg_pids[@]}" || exit 1;


mkdir -p data/lists/{aachen,original};
for c in aachen original; do
  for f in data/part/$c/*.lst; do
    bn=$(basename "$f" .lst);
    [ -s "data/lists/$c/$bn.lst" ] ||
    awk '{ print "data/imgs/lines/"$1".jpg" }' "$f" \
      > "data/lists/$c/$bn.lst" || exit 1;
    [ -s "data/lists/$c/${bn}_h${h}.lst" ] ||
    awk -v h="$height" '{ print "data/imgs/lines_h"h"/"$1".jpg" }' "$f" \
      > "data/lists/$c/${bn}_h${height}.lst" || exit 1;
  done;
done;

rm -rf "$tmpd";
exit 0;
