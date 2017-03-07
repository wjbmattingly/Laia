#!/bin/bash
set -e;
export LC_NUMERIC=C;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/utils" = "$SDIR" ] || {
    echo "Run the script from \"$(dirname $SDIR)\"!" >&2 && exit 1;
}
[ ! -f "$(pwd)/utils/parse_options.inc.sh" ] && \
    echo "Missing $(pwd)/utils/parse_options.inc.sh file!" >&2 && exit 1;

add_wspace_border=true;
eps="<eps>";
wspace="<space>";
help_message="
Usage: ${0##*/} [options] syms lkh_ark lkh_ark

Description:

Options:
  --add_wspace_border : (type = boolean, default = $add_wspace_border)
                        Add fake whitespace frame at the beginning and at the
                        end of each form.
  --eps               : (type = string, default = \"$eps\")
                        Token representing the epsilon symbol.
  --wspace            : (type = string, default \"$wspace\")
                        Token representing the whitespace character.
";
source utils/parse_options.inc.sh || exit 1;
[ $# -ne 3 ] && echo "$help_message" >&2 && exit 1;

# Parse inputs from arguments
syms="$1";
inpf="$2";
outf="$3";
scpf="${3/.ark/.scp}";

for f in "$syms" "$inpf"; do
  [ ! -s "$f" ] && echo "ERROR: File \"$f\" was not found!" >&2 && exit 1;
done;

mkdir -p "$(dirname "$outf")";

# Get the number of symbols (including CTC, but excluding epsilon) and
# the ID of the whitespace symbol.
info=( $(awk -v eps="$eps" -v ws="$wspace" '{
  if ($1 != eps) N++;
  if ($1 == ws) wss=$2;
}END{ print N, wss }' "$syms") );

copy-matrix "ark:$inpf" ark,t:- | awk -v AW="$add_wspace_border" \
  -v ND="${info[0]}" -v WSS="${info[1]}" '
  function add_dummy_frame(n) {
    printf("%s", n);
    infv=-3.4 * 10^38;
    printf(" %g", 0.0);
    for (i = 2; i <= ND; ++i) { printf(" %g", infv); }
    printf("\n");
  }
  function add_wspace_frame(n) {
    printf("%s", n);
    infv=-3.4 * 10^38;
    for (i = 1; i < WSS; ++i) { printf(" %g", infv); }
    printf(" %g", 0.0);
    for (i = WSS + 1; i <= ND; ++i) { printf(" %g", infv); }
    printf("\n");
  }
  BEGIN {
    form_id="";
  }{
    S = 1; F = NF;
    if ($2 == "[" && match($1, /^([^ ]+)-([0-9]+)$/, A)) {
      if (form_id == A[1]) {
        add_wspace_frame(form_id);
      } else if (AW == "true") {
        if(form_id != "") {
          add_wspace_frame(form_id);
          add_dummy_frame(form_id);
        }
        add_wspace_frame(A[1]);
      }
      form_id = A[1];
      S += 2;
    }
    if ($NF == "]") {
      F = NF - 1;
    }
    if (S <= F) {
      printf("%s", form_id);
      for (i = S; i <= F; ++i)
        printf(" %g", $i);
      printf("\n");
    }
  }END{
    if (AW == "true" && form_id != "") {
      add_wspace_frame(form_id);
      add_dummy_frame(form_id);
    }
  }' | awk '
  BEGIN{
    form_id="";
  }{
    if ($1 != form_id) {
      if (form_id != "") printf("]\n");
      form_id = $1;
      printf("%s [\n", form_id);
    }
    for (i = 2; i <= NF; ++i) printf(" %g", $i);
    printf("\n");
  }END{
    if (form_id != "") printf("]\n");
  }' | copy-matrix ark,t:- "ark,scp:$outf,$scpf" ||
{ echo "ERROR: Creating file \"$outf\"!" >&2 && exit 1; }

exit 0;
