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
height=64;
dataset_name="Spanish_Number_DB";

help_message="
Usage: ${0##*/} [options]

Options:
  --height     : (type = integer, default = $height)
                 Scale lines to have this height, keeping the aspect ratio
                 of the original image.
  --overwrite  : (type = boolean, default = $overwrite)
                 Overwrite previously created files.
";
source "$(pwd)/utils/parse_options.inc.sh" || exit 1;

[ -d data/$dataset_name/adq3/frases2/ -a -s data/$dataset_name/train.lst -a -s data/$dataset_name/test.lst ] || \
  ( echo "The Spanish Number database is not available!">&2 && exit 1; );

mkdir -p data/lang/{char,word};

which convert &> /dev/null || {
  echo "ImageMagick's convert was not found in your PATH" >&2;
  exit 1;
}

echo -n "Creating transcripts..." >&2;
for p in train test; do
  # Place all character-level transcripts into a single txt table file.
  # Token {space} is used to mark whitespaces between words.
  [ -s data/lang/char/$p.txt -a $overwrite = false ] || {
    for s in $(< data/$dataset_name/$p.lst); do
      echo "$s $(cat data/$dataset_name/adq3/frases2/$s.txt | sed 's/./& /g' | sed 's/@/{space}/g')";
    done > data/lang/char/$p.txt;
  }
  # Place all word-levle transcripts into a single txt table file.
  [ -s data/lang/word/$p.txt -a $overwrite = false ] || {
    for s in $(< data/$dataset_name/$p.lst); do
      echo "$s $(cat data/$dataset_name/adq3/frases2/$s.txt | tr @ \ )";
    done > data/lang/word/$p.txt;
  }
done;
echo -e "  \tDone." >&2;

echo -n "Creating symbols table..." >&2;
# Generate symbols table from training and validation characters.
# This table will be used to convert characters to integers using Kaldi format.
[ -s data/lang/char/symbs.txt -a $overwrite = false ] || (
  for p in train test; do
    cut -f 2- -d\  data/lang/char/$p.txt | tr \  \\n;
  done | sort -u -V |
  awk 'BEGIN{
    N=0;
    printf("%-12s %d\n", "<eps>", N++);
    printf("%-12s %d\n", "<ctc>", N++);
  }NF==1{
    printf("%-12s %d\n", $1, N++);
  }' > data/lang/char/symbs.txt;
)
echo -e "  \tDone." >&2;

## Resize to a fixed height and convert to png.
echo -n "Preprocessing images..." >&2;
mkdir -p data/imgs_proc;
for p in train test; do
  # [ -f data/$dataset_name/$p.lst ] && continue;
  for f in $(< data/$dataset_name/$p.lst); do
    [ -f data/imgs_proc/$f.png -a $overwrite = false ] && continue;
    [ ! -f data/$dataset_name/adq3/frases2/$f.pbm ] && \
      echo "Image data/$dataset_name/adq3/frases2/$f.pbm is not available!">&2 \
        && exit 1;
      #echo "File data/$dataset_name/adq3/frases2/$f.pbm..." >&2;
      convert -interpolative-resize "x$height" data/$dataset_name/adq3/frases2/$f.pbm data/imgs_proc/$f.png
  done;
  awk '{ print "data/imgs_proc/"$1".png" }' data/$dataset_name/$p.lst > data/$p.lst;
done;
echo -e "  \tDone." >&2;

exit 0;
