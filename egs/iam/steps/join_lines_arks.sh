#!/bin/bash
export LC_NUMERIC=C;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/steps" = "$SDIR" ] || {
    echo "Run the script from \"$(dirname $SDIR)\"!" >&2 && exit 1;
}
[ ! -f "$(pwd)/utils/parse_options.inc.sh" ] && \
    echo "Missing $(pwd)/utils/parse_options.inc.sh file!" >&2 && exit 1;

eps="<eps>";
overwrite=false;
wspace="<space>";
help_message="
Usage: ${0##*/} [options]

Description:

Options:
  --eps           : (type = string, default = \"$eps\")
                    Token representing the epsilon symbol.
  --overwrite     : (type = int, default = $overwrite)
                    Overwrite existing files from previous runs.
  --wspace        : (type = string, default \"$wspace\")
                    Token representing the whitespace character.
";
source utils/parse_options.inc.sh || exit 1;
[ $# -ne 0 ] && echo "$help_message" >&2 && exit 1;
# Expected inputs
syms="train/lines/syms.txt";
inp_va="decode/no_lm/char/lines/aachen/va_lstm1d_h128.ark";
inp_te="decode/no_lm/char/lines/aachen/te_lstm1d_h128.ark";
# Check that expected inputs exist
for f in "$syms" "$inp_va" "$inp_te"; do
  [ ! -s "$f" ] && echo "ERROR: File \"$f\" was not found!" >&2 && exit 1;
done;
# Create output directory, if it does not exist
mkdir -p decode/no_lm/char/forms/aachen;
# Output variables
out_va="decode/no_lm/char/forms/aachen/va_lstm1d_h128.ark";
out_te="decode/no_lm/char/forms/aachen/te_lstm1d_h128.ark";

# Get the number of symbols (including CTC, but excluding epsilon) and
# the ID of the whitespace symbol.
info=( $(awk -v eps="$eps" -v ws="$wspace" '{
  if ($1 != eps) N++;
  if ($1 == ws) wss=$2;
}END{ print N, wss }' "$syms") );

tmpf="$(mktemp)";
inps=("$inp_va" "$inp_te");
outs=("$out_va" "$out_te");
for i in $(seq 1 ${#inps[@]}); do
  [ "$overwrite" = false -a -s "${outs[i-1]}" ] ||
  copy-matrix "ark:${inps[i-1]}" "ark,scp:$tmpf,-" |
  awk -v tmpf="$tmpf" -v inpf="${inps[i-1]}" \
  '{ gsub(tmpf, inpf); print; }' | sort |
  copy-matrix scp:- ark,t:- | awk -v ND="${info[0]}" -v WSS="${info[1]}" '
  BEGIN {
    form_id=""; infv=-999;
  }{
    S = 1; F = NF;
    if ($2 == "[" && match($1, /^([^ ]+)-([0-9]+)$/, A)) {
      if (form_id == A[1]) {
        # We put an auxiliar whitespace frame, with all the mass concentrated
        # into the whitespace symbol.
        printf("%s", form_id);
        for (i = 1; i < WSS; ++i) { printf(" %g", infv); }
        printf(" %g", 0.0);
        for (i = WSS + 1; i <= ND; ++i) { printf(" %g", infv); }
        printf("\n");
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
  }' | copy-matrix ark,t:- "ark:${outs[i-1]}" ||
  { echo "ERROR: Creating file \"${outs[i-1]}\"!" >&2 && exit 1; }
done;

rm "$tmpf";
exit 0;
