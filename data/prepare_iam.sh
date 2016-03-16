#!/bin/bash
set -e;

# Directory where the prepare_iam.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "$SDIR";

[ -d iam/gt -a -d iam/imgs -a -f iam/train.lst ] || \
    ( echo "The IAM database is not available in data/iam!">&2 && exit 1; );

for p in train; do
    # Place all character-level transcriptions into a single txt table file.
    # The original transcriptions may contain some contractions already
    # splitted (e.g. he 's), merge those contractions in order to avoid
    # artificial whitespaces.
    # Token <space> is used to mark whitespaces between words.
    # Also, replace # by <stroke>.
    for f in $(< iam/$p.lst); do
        echo -n "$f "; zcat iam/gt/$f.txt.gz | \
	    sed 's/ '\''\(s\|d\|ll\|ve\|t\|re\|S\|D\|LL\|VE\|T\|RE\)\([ $]\)/'\''\1\2/g' | \
            awk '{
          for(i=1;i<=NF;++i) {
            for(j=1;j<=length($i);++j)
              printf(" %s", substr($i, j, 1));
            if (i < NF) printf(" <space>");
          }
          printf("\n");
        }' | tr \` \' | sed 's/"/'\'' '\''/g' | sed 's/#/<stroke>/g';
    done > iam_${p}.txt;
done;
