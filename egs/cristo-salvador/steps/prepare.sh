#!/bin/bash
set -e;

# Directory where the prepare.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/steps" != "$SDIR" ] && \
    echo "Please, run this script from the experiment top directory!" && \
    exit 1;
[ ! -f "$(pwd)/utils/parse_options.inc.sh" ] && \
    echo "Missing $(pwd)/utils/parse_options.inc.sh file!" >&2 && exit 1;

overwrite=false;
height=64;
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

[ -d data/corpus/Line-Level -a -s data/corpus/train_book.lst \
  -a -s data/corpus/test_book.lst ] || \
  ( echo "The CS database is not available!">&2 && exit 1; );

for tool in imgtxtenh imageSlant convert; do
  which "$tool" > /dev/null || \
    (echo "Required tool $tool was not found!" >&2 && exit 1);
done;

mkdir -p data/lang/{char,word};

for p in train test; do
  [ -f data/lang/word/$p.orig.txt -a \
    -f data/lang/word/$p.expanded.txt -a \
    -f data/lang/char/$p.txt -a "$overwrite" = false ] && continue;
  # Convert from ISO-8859-1 to UTF-8 and put all transcripts in a single
  # file.
  for f in $(tail -n+2 data/corpus/${p}_book.lst); do
    echo -n "$f ";
    sed -r 's|[ \t]+| |g;s|^ ||g;s| $||g;s|_| |g' data/corpus/Line-Level/$f.txt;
  done | sort -V > data/lang/word/$p.orig.txt;
  # Write words fully expanded, without the expansion mark symbols:
  # S[eño]r -> Señor ; D[octo]r -> Doctor
  sed -r 's/(\[|\]|\<|\>)//g' data/lang/word/$p.orig.txt \
    > data/lang/word/$p.expanded.txt;
  # Write words as they are written in the text lines:
  # S[eño]r -> Sr. ; D[octo]r -> Dr.
  # <él>  -> ;
  # For the character-level transcripts, use @ as the whitespace symbol.
  sed -r 's/(\S*\[\S*\]\S*)/\1./g;s/\[\S*\]//g;s/<\S*>//g' \
    data/lang/word/$p.orig.txt | awk '{
      printf("%s", $1);
      for (i=2; i<=NF; ++i) {
        for(j=1;j<=length($i);++j) {
          printf(" %s", substr($i, j, 1));
        }
        if (i < NF) printf(" @");
      }
      printf("\n");
    }' > data/lang/char/$p.txt;
done;

# Generate symbols table from training and valid characters.
# This table will be used to convert characters to integers by Kaldi and the
# CNN + RNN + CTC code.
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

## Enhance images with Mauricio's tool, crop image white borders and resize
## to a fixed height.
mkdir -p data/imgs_proc;
bkg_pids=();
bkg_errs=();
np="$(nproc)";
for f in $(awk '{print $1}' data/lang/char/train.txt \
  data/lang/char/test.txt); do
  finp=data/corpus/Line-Level/$f.png;
  fout=data/imgs_proc/$f.jpg;
  [ -f "$fout" -a $overwrite = false ] && continue;
  [ ! -f "$finp" ] && echo "Image $finp is not available!">&2 && exit 1;
  (
    echo "File $finp..." >&2;
    imgtxtenh -d 118.1102362205 -k0.1 -S20  "$finp" "${fout}";
    trim1="$(convert "${fout}" +repage -fuzz 5% -trim \
            -print '%@' +repage "${fout}")";
    blackb="$(python $SDIR/remove_black_border.py "${fout}")";
    convert "${fout}" -crop "$blackb" +repage "${fout}";
    slope="$(convert "${fout}" +repage -flatten -deskew 40% \
            -print '%[deskew:angle]\n' +repage "${fout}")";
    slant="$(imageSlant -v 1 -g -i "${fout}" -o "${fout}" \
            2>&1 | sed -n '/Slant medio/{s|.*: ||;p;}')";
    trim2="$(convert "${fout}" +repage -fuzz 5% -trim -print '%@' \
            +repage "${fout}")";
    convert "${fout}" -resize "x$height" -strip +repage "$fout";
    echo "Remove white borders1: $trim1";
    echo "Remove black borders: $blackb";
    echo "Slope: $slope";
    echo "Slant: $slant";
    echo "Remove white borders2: $trim2";
  ) &> data/imgs_proc/$f.log &
  bkg_pids+=("$!");
  bkg_errs+=("data/imgs_proc/$f.log");
  if [ "${#bkg_pids[@]}" -eq "$np" ]; then
    for n in $(seq 1 "${#bkg_pids[@]}"); do
      wait "${bkg_pids[n-1]}" || (
        echo "Failed image processing step:" >&2 && \
          cat "${bkg_errs[n-1]}" >&2 && exit 1;
      );
    done;
    bkg_pids=();
  fi;
done;


## Prepare test, train and valid files.
awk '{ print "data/imgs_proc/"$1".jpg"; }' data/lang/char/test.txt \
  > data/test.lst;
TMPF="$(mktemp)";
sort -R --random-source=data/lang/char/train.txt data/lang/char/train.txt |
awk '{ print "data/imgs_proc/"$1".jpg"; }' > "$TMPF";
head -n100  "$TMPF" | sort -V > data/valid.lst;
tail -n+101 "$TMPF" | sort -V > data/train.lst;
rm -f "$TMPF";

exit 0;
