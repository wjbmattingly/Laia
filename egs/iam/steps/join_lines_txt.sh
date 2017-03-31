#!/bin/bash
set -e;
export LC_NUMERIC=C;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/steps" = "$SDIR" ] || {
    echo "Run the script from \"$(dirname $SDIR)\"!" >&2 && exit 1;
}
[ ! -f "$(pwd)/utils/parse_options.inc.sh" ] && \
    echo "Missing $(pwd)/utils/parse_options.inc.sh file!" >&2 && exit 1;

help_message="
Usage: ${0##*/} [options] wspace input_txt

Description:

Options:
  --eps               : (type = string, default = \"$eps\")
                        Token representing the epsilon symbol.
  --wspace            : (type = string, default \"$wspace\")
                        Token representing the whitespace character.
";
source utils/parse_options.inc.sh || exit 1;
[ $# -ne 2 ] && echo "$help_message" >&2 && exit 1;

# Parse inputs from arguments
wspace="$1";
inpf="$2";

[ ! -s "$inpf" ] && echo "ERROR: File \"$inpf\" was not found!" >&2 && exit 1;

sort -V "$inpf" | awk -v ws="$wspace" '
BEGIN {
  form_id="";
}{
  if (!match($1, /^([^ ]+)-([0-9]+)$/, A)) {
    print "IGNORING LINE "NR": "$0 > "/dev/stderr"; next;
  }
  if (form_id == A[1]) {
    if (ws != "") printf(" %s", ws);
  } else {
    if (form_id != "") printf("\n");
    printf("%s", A[1]);
    form_id = A[1];
  }
  for (i = 2; i <= NF; ++i) {
    printf(" %s", $i);
  }
}END{
  if (form_id != "") {
    printf("\n");
  }
}'

exit 0;
