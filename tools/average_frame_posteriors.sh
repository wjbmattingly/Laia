#!/bin/bash
export LC_NUMERIC=C
set -e

if [ $# -lt 5 ]; then
    usage="
Usage: ${0##*/} samplesLst numSymb outDir pst1Dir pst2Dir [pst3Dir ...]

Average the frame posteriors of a set of models.

For each sample, a file named as the sample ID with the .fea extension must be
placed in a different directories, one for each model to average.

It is expected that all models produce the same number of frames for each
sample (rows in the original .fea files) and that all were trained with the
same number of output labels (columns in the original .fea files).
"
    echo "$usage" >&2;
    exit 1;
fi;

LST="$1";
LABELS="$2";
OUTD="$3";
shift 3;

[ ! -f "$LST" ] && echo "File $LST does not exist!" >&2 && exit 1;
[ "$LABELS" -le 0 ] && echo "Number of labels must be positive!" >&2 && exit 1;
mkdir -p "$OUTD" || ( echo "Failed to create dir $OUTD!" >&2 && exit 1; )

for f in $(< "$LST"); do
    echo "$f" >&2;
    mapfile -t A <<< "$(for d in "$@"; do echo "$d/$f.fea"; done)";
    paste "${A[@]}" | awk -v F="$f" -v L="$LABELS" -v M="$#" '{
      if (NF != L * M) {
        print "Some frames for sample "F" do not have the appropiate number of columns (expected = "L * M", actual ="NF")" > "/dev/stderr";
        exit(1);
      }
      for (l = 0; l < L; ++l) {
        S[l] = 0;
        for (m=0; m < M; ++m) { S[l] += $(m * L + l + 1); }
        printf("%.8f ", S[l]);
      }
      printf("\n");
    }' > "$OUTD/$f.fea";
done;
