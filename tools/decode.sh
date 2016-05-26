#!/bin/bash
set -e;
export LC_NUMERIC=C;

# Directory where the prepare_iam.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";

symbols_table="";
help_message="
Usage: ${0##*/} [options] model data

Options:
  --symbols_table  : (type = string, default = \"$symbols_table\")
                     File containing the mapping from symbols to integers.
";
. "${SDIR}/parse_options.inc.sh" || exit 1;
[ $# -eq  0 ] && echo "$help_message" >&2 && exit 1;
[ $# -ne 2 ] && echo "Wrong number of arguments, see --help" >&2 && exit 1;

model="$(realpath "$1")";
data="$(realpath "$2")";
symbols_table="$(realpath "$symbols_table")";

#cd "$SDIR/..";
th decode.lua "$model" "$data" | \
    sed -r 's|([0-9]+)( \1)+|\1 |g;s| 0||g;s|\ +| |g' | \
    awk -v STF="$symbols_table" '
BEGIN{
  if (STF != "") { while ((getline < STF) > 0) ST[$2]=$1; }
}{
  for(i=2;i<=NF;++i) { if ($i in ST) $i=ST[$i]; }
  print;
}' || exit 1;
exit 0;
