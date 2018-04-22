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


acoustic_scale=1;
allow_partial=true;
beam=30;
max_active=5000000;
num_procs="$(nproc)";
order=3;
overwrite=false;
help_message="
Usage: ${0##*/} [options]
Options:
  --order        : (type = integer, default = $order)
                   Batch size for Laia.
  --overwrite    : (type = boolean, default = $overwrite)
                   Overwrite previous files.
";
source utils/parse_options.inc.sh || exit 1;


for cv in cv1 cv2 cv3 cv4; do
  for p in te tr va; do
    [ "$overwrite" = false -a \
      -s "data/lang/word/$cv/${p}_lexiconp_count.txt" -a \
      -s "data/lang/word/$cv/${p}_tokenized.txt" ] ||
    gawk -v LEXC=data/lang/word/$cv/${p}_lexiconp_count.txt '
    function print_word(w, p, cnt) {
      if (w != "") {
        printf(" %s", w);
        cnt[w"___"p]++;
      }
    }
    BEGIN{

    }{
      printf("%s", $1);
      w = "";
      p = "<space>";
      for (i = 2; i <= NF; ++i) {
        if ($i == "s_bl") {
          print_word(w, p, CNT);
          print_word("(", $i, CNT);
          w = "";
          p = "";
        } else if ($i == "s_br") {
          print_word(w, p, CNT);
          print_word(")", $i, CNT);
          w = "";
          p = "";
        } else if ($i == "s_cm") {
          print_word(w, p, CNT);
          print_word(",", $i, CNT);
          w = "";
          p = "";
        } else if ($i == "s_et") {
          w = sprintf("%sV", w);
          p = sprintf("%s %s", p, $i);
        } else if ($i == "s_lb") {
          w = sprintf("%sL", w);
          p = sprintf("%s %s", p, $i);
        } else if ($i == "s_mi") {
          w = sprintf("%s-", w);
          p = sprintf("%s %s", p, $i);
        } else if ($i == "s_pt") {
          print_word(w, p, CNT);
          print_word(".", $i, CNT);
          w = "";
          p = "";
        } else if ($i == "s_qo") {
          print_word(w, p, CNT);
          print_word(":", $i, CNT);
          w = "";
          p = "";
        } else if ($i == "s_qt") {
          w = sprintf("%s'\''", w);
          p = sprintf("%s %s", p, $i);
        } else if ($i == "s_s")  {
          w = sprintf("%ss", w);
          p = sprintf("%s %s", p, $i);
        } else if ($i == "s_sl") {
          w = sprintf("%ssl", w);
          p = sprintf("%s %s", p, $i);
        } else if ($i == "s_sq") {
          print_word(w, p, CNT);
          print_word(";", $i, CNT);
          w = "";
          p = "";
        } else if (match($i, /^s_(.+)$/, A)) {
          w = sprintf("%s%s", w, A[1]);
          p = sprintf("%s %s", p, $i);
        } else if ($i == "<space>") {
          print_word(w, p, CNT);
          w = "";
          p = "<space>";
        } else {
          w = sprintf("%s%s", w, $i);
          p = sprintf("%s %s", p, $i);
        }
      }
      if (w != "") {
        print_word(w, p, CNT);
      }
      printf("\n");
    }END{
      for (wp in CNT) {
        split(wp, WP, "___");
        printf("%-15s %d %s\n", WP[1], CNT[wp], WP[2]) > LEXC;
      }
    }' data/lang/char/$cv/$p.txt > data/lang/word/$cv/${p}_tokenized.txt;
  done;

  [ "$overwrite" = false -a -s "data/lang/word/$cv/all_lexiconp_count.txt" ] ||
  awk '{
    w=$1;
    p=$3;
    for(i=4;i<=NF;++i) p = sprintf("%s %s", p, $i);
    CNT[w"___"p] += $2;
  }END{
    for (wp in CNT) {
      split(wp, WP, "___");
      printf("%-15s %d %s\n", WP[1], CNT[wp], WP[2]);
    }
  }' "data/lang/word/$cv"/{te,tr,va}_lexiconp_count.txt |
  sort -V > data/lang/word/$cv/all_lexiconp_count.txt;

  [ "$overwrite" = false -a -s "data/lang/word/$cv/all_voc.txt" ] ||
  cut -d\  -f1 data/lang/word/$cv/all_lexiconp_count.txt |
    sort -V > data/lang/word/$cv/all_voc.txt;


  [ "$overwrite" = false -a -s "data/lang/word/$cv/all_lexiconp.txt" ] ||
  gawk '{
    w=$1;
    p=$3;
    for(i=4;i<=NF;++i) p = sprintf("%s %s", p, $i);
    CNT[w"___"p] += $2;
    CNTW[w] += $2;
  }END{
    for (wp in CNT) {
      split(wp, WP, "___");
      printf("%-15s %.4f %s\n", WP[1], CNT[wp] / CNTW[WP[1]], WP[2]);
    }
  }' data/lang/word/$cv/all_lexiconp_count.txt |
  sort -V > data/lang/word/$cv/all_lexiconp.txt;
done;


for cv in cv1; do #cv1 cv2 cv3 cv4; do
  mkdir -p "decode/lm/$cv";
  # Build ARPA
  [ "$overwrite" = false -a -s "decode/lm/$cv/lm.arpa" ] ||
  ngram-count -order "$order" -wbdiscount -interpolate \
	      -text <(cut -d\  -f2- data/lang/word/$cv/tr_tokenized.txt) \
	      -vocab "data/lang/word/$cv/all_voc.txt" \
	      -lm "decode/lm/$cv/lm.arpa";
  echo -n "$cv ";
  ngram -order "$order" -ppl <(cut -d\  -f2- data/lang/word/$cv/va_tokenized.txt) \
	-lm "decode/lm/$cv/lm.arpa" | tr \\n  \  |
  sed -r 's|^.* ([0-9]+) words, ([0-9]+) OOVs.* ppl= ([0-9.]+) .*$|ppl = \3 ; oov = \2/\1\n|g';

  # Prepare FSTs
  ./utils/build_word_fsts.sh \
    --overwrite "$overwrite" \
    data/lang/char/syms.txt \
    data/lang/word/$cv/all_lexiconp.txt \
    "decode/lm/$cv/lm.arpa" \
    "decode/lm/$cv/fsts";

  # Launch decoding
  for p in va; do
    ./utils/decode_lazy.sh \
      --acoustic_scale "$acoustic_scale" \
      --beam "$beam" \
      --max_active "$max_active" \
      --num_procs "$num_procs" \
      --overwrite "$overwrite" \
      --symbol_table "decode/lm/$cv/fsts/words.txt" \
      "decode/lm/$cv/fsts"/{model,HCL.fst,G.fst} \
      "decode/lkh/$cv/${p}_space.scp" \
      "decode/lm/$cv/fsts/$p";

    char_txt="decode/lm/$cv/char/$p.txt";
    mkdir -p "decode/lm/$cv/char";
    [ "$overwrite" = false -a -s "$char_txt" ] ||
      for f in "decode/lm/$cv/fsts/$p"/align.*.of.*.ark.gz; do
	ali-to-phones "decode/lm/$cv/fsts/model" \
		      "ark:zcat $f|" ark,t:- 2> /dev/null
      done |
      ./utils/int2sym.pl -f 2- "decode/lm/$cv/fsts/chars.txt" |
      ./utils/remove_transcript_dummy_boundaries.sh > "$char_txt";

    word_txt="decode/lm/$cv/word/$p.txt";
    mkdir -p "decode/lm/$cv/word";
    [ "$overwrite" = false -a -s "$word_txt" ] ||
      ./utils/remove_transcript_dummy_boundaries.sh \
	--to-words "$char_txt" > "$word_txt";
  done;
done;


echo "";
for cv in cv1; do # cv2 cv3 cv4; do
  echo -n "$cv Valid ";
  compute-wer-bootci --mode=present --print-args=false \
    ark:data/lang/word/$cv/va_tokenized.txt ark:decode/lm/$cv/word/va.txt |
    grep WER | sed -r 's|Set1: ||g;s|95% Conf Interval ||g';

#  echo -n "$cv Test ";
#  compute-wer-bootci --mode=strict --print-args=false \
#    ark:data/lang/word/$cv/te.txt ark:decode/no_lm/word/$cv/te.txt |
#    grep WER | sed -r 's|Set1: ||g;s|95% Conf Interval ||g';
done;
