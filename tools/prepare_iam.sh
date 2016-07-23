#!/bin/bash
set -e;

# Directory where the prepare_iam.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/tools" != "$SDIR" ] && \
    echo "Please, run this script from the project top directory!" && \
    exit 1;

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
. "${SDIR}/parse_options.inc.sh" || exit 1;

[ -d data/iam/gt -a -d data/iam/imgs -a -s data/iam/train.lst ] || \
    ( echo "The IAM database is not available in data/iam!">&2 && exit 1; );

mkdir -p data/iam/lang/chars;

for p in train valid test; do
    # Place all character-level transcriptions into a single txt table file.
    # The original transcriptions may contain some contractions already
    # splitted (e.g. he 's), merge those contractions in order to avoid
    # artificial whitespaces.
    # Token <space> is used to mark whitespaces between words.
    # Also, replace # by <stroke>.
    [ -s data/iam/lang/chars/$p.txt -a $overwrite = false ] && continue;
    for f in $(< data/iam/$p.lst); do
        echo -n "$f "; zcat data/iam/gt/$f.txt.gz | \
	    sed 's/ '\''\(s\|d\|ll\|ve\|t\|re\|S\|D\|LL\|VE\|T\|RE\)\([ $]\)/'\''\1\2/g' | \
            awk '{
          for(i=1;i<=NF;++i) {
            for(j=1;j<=length($i);++j)
              printf(" %s", substr($i, j, 1));
            if (i < NF) printf(" <space>");
          }
          printf("\n");
        }' | tr \` \' | sed 's/"/'\'' '\''/g' | sed 's/#/<stroke>/g';
    done > data/iam/lang/chars/$p.txt;
done;

# Generate symbols table from training and valid characters.
# This table will be used to convert characters to integers by Kaldi and the
# CNN + RNN + CTC code.
[ -s data/iam/lang/chars/original_symbols.txt -a $overwrite = false ] || (
    for p in train valid; do
	cut -f 2- -d\  data/iam/lang/chars/$p.txt | tr \  \\n;
    done | sort -u -V | \
	awk 'BEGIN{N=1;}NF==1{ printf("%-10s %d\n", $1, N); N++; }' > \
	data/iam/lang/chars/original_symbols.txt;
)

# Prepare data for Torch7
# Image colors are inverted: 0->white, 1->black.
# Other preprocessing worth trying: normalization (mean = 0, stddev = 1), ZCA
for p in train valid test; do
    [ -s data/iam/$p.h5 -a $overwrite = false ] || \
	th "$SDIR/create_hdf5.lua" --invert true --height "$height" \
	data/iam/lang/chars/$p.txt data/iam/lang/chars/original_symbols.txt \
	data/iam/imgs data/iam/$p.h5;
done;

exit 0;
