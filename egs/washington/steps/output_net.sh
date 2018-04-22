#!/bin/bash
set -e;
export LC_NUMERIC=C;
export LUA_PATH="$(pwd)/../../?/init.lua;$(pwd)/../../?.lua;$LUA_PATH";

# Directory where the prepare.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/steps" != "$SDIR" ] && \
  echo "Please, run this script from the experiment top directory!" >&2 && \
  exit 1;
[ ! -f "$(pwd)/utils/parse_options.inc.sh" ] && \
  echo "Missing $(pwd)/utils/parse_options.inc.sh file!" >&2 && exit 1;

batch_size=16;
overwrite=false;
prior_scale=1.0;
help_message="
Usage: ${0##*/} [options]
Options:
  --batch_size      : (type = integer, default = $batch_size)
                      Batch size for Laia.
  --overwrite       : (type = boolean, default = $overwrite)
                      Overwrite previous files.
";
source utils/parse_options.inc.sh || exit 1;

# Check required files
for cv in cv1 cv2 cv3 cv4; do
  for f in "data/lang/char/syms.txt" \
           "train/$cv/model.t7" \
           "data/lists/$cv/te.txt" \
	   "data/lists/$cv/tr.txt" \
           "data/lists/$cv/va.txt"; do
    [ ! -s "$f" ] && echo "ERROR: File \"$f\" was not found!" >&2 && exit 1;
  done;
done;

# Compute log-posteriors from the network.
for cv in cv1 cv2 cv3 cv4; do
  mkdir -p decode/pst/$cv;
  for p in te tr va; do
    ark="decode/pst/$cv/$p.ark";
    scp="decode/pst/$cv/$p.scp";
    [ "$overwrite" = false -a -s "$ark" -a -s "$scp" ] ||
      ../../laia-netout \
	--batch_size "$batch_size" \
	--output_transform "logsoftmax" \
	"train/$cv/model.t7" \
	"data/lists/$cv/$p.txt" \
	/dev/stdout |
      copy-matrix ark:- "ark,scp:$ark,$scp";
  done;
done;

# Estimate label priors
for cv in cv1 cv2 cv3 cv4; do
  ark="decode/pst/$cv/tr.ark";
  [ "$overwrite" = false -a -s decode/pst/$cv/tr.prior ] ||
  copy-matrix "ark:$ark" ark,t:- |
  gawk -v NL="$[$(wc -l data/lang/char/syms.txt | cut -d\  -f1) - 1]" '
  function logadd(x, y) {
    #if (y > x) { t=x; x = y; y = t; }
    return x + log(1.0 + exp(y - x));
  }NF==NL{
    N++;
    for (i = 1; i <= NF; ++i) {
      if(i in PRIOR) {
        PRIOR[i] = logadd(PRIOR[i], $i);
      } else {
        PRIOR[i] = $i;
      }
    }
  }END{
    for (i = 1; i <= NL; ++i) {
      printf("%-3d %.8g\n", i, PRIOR[i] - log(N));
    }
  }' > decode/pst/$cv/tr.prior;
done;

for cv in cv1 cv2 cv3 cv4; do
  mkdir -p decode/lkh/$cv;
  for p in te tr va; do
    inp="decode/pst/$cv/$p.ark";
    out="decode/lkh/$cv/${p}_ps${prior_scale}.ark";
    [ "$overwrite" = false -a -s "$out" ] ||
    copy-matrix "ark:$inp" ark,t:- |
    gawk -v PF=decode/pst/$cv/tr.prior -v PS="$prior_scale" '
    BEGIN{
      while ((getline < PF) > 0) {
        PRIOR[$1] = $2;
        ++NL;
      }
    }{
      if (NF >= NL) {
        for (i = 1; i <= NL; ++i) {
          $i = $i - PS * PRIOR[i];
        }
      }
      print;
    }' | copy-matrix ark:- "ark:$out";
  done;
done;

# Add fake frames at the start and end of the line with whitespace symbol.
for cv in cv1 cv2 cv3 cv4; do
  for p in te va; do
    inp="decode/pst/$cv/$p.ark";
    out="decode/pst/$cv/${p}_space";
    [ "$overwrite" = false -a -s "$out.ark" -a -s "$out.scp" ] ||
      ./utils/add_border_fake_frames.sh "<space>" data/lang/char/syms.txt \
					"ark:$inp" "ark,scp:$out.ark,$out.scp";

    inp="decode/lkh/$cv/$p.ark";
    out="decode/lkh/$cv/${p}_space";
    [ "$overwrite" = false -a -s "$out.ark" -a -s "$out.scp" ] ||
      ./utils/add_border_fake_frames.sh "<space>" data/lang/char/syms.txt \
					"ark:$inp" "ark,scp:$out.ark,$out.scp";
  done;
done;
