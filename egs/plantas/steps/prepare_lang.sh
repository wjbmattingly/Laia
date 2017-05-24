#!/bin/bash
set -e;
export LC_NUMERIC=C;

# Directory where this script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/steps" != "$SDIR" ] && \
    echo "Please, run this script from the experiment top directory!" >&2 && \
    exit 1;
[ ! -f "$(pwd)/utils/parse_options.inc.sh" ] && \
    echo "Missing $(pwd)/utils/parse_options.inc.sh file!" >&2 && exit 1;

overwrite=false;
help_message="
Usage: ${0##*/} [options]

Options:
  --overwrite
";
source utils/parse_options.inc.sh || exit 1;

for f in data/lang/word/va_original.txt data/lang/word/te_original.txt; do
  [ ! -f "$f" ] && echo "File \"$f\" does not exist!" >&2 && exit 1;
done;

# We need to remove <add>(.+)</add> from the text, since this text is actually
# present in a different line.
function remove_add () {
  awk '{
    if (match($0, "(.+)<add>(.+)</add>(.*)", A)) {
      print A[1], A[3];
    } else {
      print $0;
    }
  }'
}

# We need to remove the <unclear> tags.
function remove_unclear () {
  sed -r 's|<unclear>||g;s|</unclear>||g;'
}

# Remove special tags of words.
function remove_tags () {
  awk '{
    printf("%s", $1);
    for (i = 2; i <= NF; ++i) {
      if (match($i, /^(.+)\$\.(.+)$/, A)) {     # Abbrev
        printf(" %s", A[1]);
      } else if (match($i, /^\$l:(.+)$/, A)) {  # Latin tag
        printf(" %s", A[1]);
      } else if (match($i, /^\$g:(.+)$/, A)) {  # Greek tag
        printf(" %s", A[1]);
      } else if (match($i, /^\$eg:(.+)$/, A)) { # Greek tag 2 (?)
        printf(" %s", A[1]);
      } else if (match($i, /^\$f:(.+)$/, A)) {  # French tag
        printf(" %s", A[1]);
      } else if (match($i, /^\$i:(.+)$/, A)) {  # Italian tag
        printf(" %s", A[1]);
      } else if (match($i, /^\$ge:(.+)$/, A)) { # German tag
        printf(" %s", A[1]);
      } else if (match($i, /^\$a:(.+)$/, A)) {  # German tag 2 (?)
        printf(" %s", A[1]);
      } else if (match($i, /^\$fl:(.+)$/, A)) { # Dutch tag (?)
        printf(" %s", A[1]);
      } else if (match($i, /^\$po:(.+)$/, A)) { # Portuguese tag (?)
        printf(" %s", A[1]);
      } else if (match($i, /^\$p:(.+)$/, A)) {  # Portuguese tag 2 (?)
        printf(" %s", A[1]);
      } else if (match($i, /^\$h:(.+)$/, A)) {  # Hebrew tag
        printf(" %s", A[1]);
      } else if (match($i, /^\$e:(.+)$/, A)) {  # English tag
        printf(" %s", A[1]);
      } else if (match($i, /^\$c:(.+)$/, A)) {  # Catalan tag
        printf(" %s", A[1]);
      } else if (match($i, /^\$m:(.+)$/, A)) {  # Maltese tag
        printf(" %s", A[1]);
      } else if (match($i, /^\$s:(.+)$/, A)) {  # Unknown lang tag 1
        printf(" %s", A[1]);
      } else if (match($i, /^\$b:(.+)$/, A)) {  # Unknown lang tag 2
        printf(" %s", A[1]);
      } else if (match($i, /^\$hu:(.+)$/, A)) { # Unknown lang tag 3
        printf(" %s", A[1]);
      } else if (match($i, /^\$v:(.+)$/, A)) {  # Unknown lang tag 4
        printf(" %s", A[1]);
      } else if (match($i, /^\$:(.+)$/, A)) {   # Unknown language
        printf(" %s", A[1]);
      } else if (match($i, /^\$-(.+)$/, A)) {   # Broken word, start of line
        printf(" %s", A[1]);
      } else if (match($i, /^(.+)\$-$/, A)) {   # Broken word, end of line
        printf(" %s", A[1]);
      } else if (match($i, /^\$\/(.+)$/, A)) {  # Crossed word
        printf(" %s", A[1]);
      } else if (match($i, /^\$>(.+)$/, A)) {   # Word aligned to the right
        printf(" %s", A[1]);
      } else if (match($i, /^\$_(.+)$/, A)) {   # Unerlined word
        printf(" %s", A[1]);
      } else if (match($i, /^\$[a-z][a-z]?:$/)) {  # Weird alphabet
        printf(" %%");
      } else if (match($i, /^(.+)\$\^(.+)$/, A)) { # Upper-score
        printf(" %s%s", A[1], A[2]);
      } else {
        printf(" %s", $i);
      }
    }
    printf("\n");
  }'
}

# Obtain diplomatic word-level transcription of the validation and test lines
# (i.e. what is really written in the image).
for f in data/lang/word/{va,te}_original.txt; do
  f2="${f/_original/_diplomatic}";
  [ "$overwrite" = false  -a  -f "$f2" ] && continue;
  cat "$f" | remove_add | remove_unclear |
  # remove_tags is used many times to remove nested tags.
  remove_tags | remove_tags | remove_tags | remove_tags |
  sed -r 's| +| |g' > "$f2";
done;

# Obtain the diplomat char-level transcription.
for f in data/lang/word/{va,te}_diplomatic.txt; do
  f2="data/lang/char/$(basename "$f")";
  [ "$overwrite" = false  -a  -f "$f2" ] && continue;
  awk '{
    printf("%s", $1);
    for (i=2; i <= NF; ++i) {
      for (j = 1; j <= length($i); ++j) {
        printf(" %s", substr($i, j, 1));
      }
      if (i < NF) printf(" @");
    }
    printf("\n");
  }' "$f" > "$f2";
done;
