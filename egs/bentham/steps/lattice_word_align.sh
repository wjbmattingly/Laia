#!/bin/bash
set -e;
export LC_NUMERIC=C;

# Directory where the prepare.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/steps" != "$SDIR" ] && \
    echo "Please, run this script from the experiment top directory!" >&2 && \
    exit 1;
[ ! -f "$(pwd)/utils/parse_options.inc.sh" ] && \
    echo "Missing $(pwd)/utils/parse_options.inc.sh file!" >&2 && exit 1;

overwrite=false;
prior_scale=0.2;
convert_to_slf=false;
help_message="
Usage: ${0##*/} [options]

Options:
";
source utils/parse_options.inc.sh || exit 1;
[ $# -ne 0 ] && echo "$help_message" >&2 && exit 1;

for f in data/lang/word/Bentham-Batch1_newTok_newPhones.dic \
         decode/lm/original_lm/model \
         decode/lm/original_lm/chars.txt \
         decode/lm/original_lm/words.txt; do
  [ ! -f "$f" ] && echo "ERROR: File \"$f\" does not exist!" >&2 && exit 1;
done;


[[ -f decode/lm/original_lm/word_align_lexicon.txt &&
   -f decode/lm/original_lm/words2.txt &&
   "$overwrite" = "false" ]] ||
./steps/htk_dictionary_to_word_align_lexicon.sh \
  data/lang/word/Bentham-Batch1_newTok_newPhones.dic \
  decode/lm/original_lm/chars.txt \
  decode/lm/original_lm/words.txt \
  decode/lm/original_lm/words2.txt \
  decode/lm/original_lm/word_align_lexicon.txt ||
exit 1;


for p in va te; do
  latdir="decode/lm/original_lm/lattices/${p}_ps${prior_scale}"
  mapfile -t lats <<< "$(find "$latdir" -maxdepth 1 -name "*.ark.gz")";
  [ "${#lats[@]}" -eq 0 ] &&
  echo "ERROR: No lattices found in \"$latdir\"!" >&2 && exit 1;
  mkdir -p "$latdir/aligned";
  for ilat in "${lats[@]}"; do
    olat="$latdir/aligned/$(basename "$ilat")";
    [[ -f "$olat" && "$overwrite" = false ]] && continue;
    # NOTE: lattice-align-words-lexicon will fail because our G.fst
    # transducer removed the end-of-sentence symbols (i.e. </s>) from
    # the output (i.e. it was subtituted by an epsilon). Thus, the
    # frames originally aligned to this "word" cannot be aligned correctly,
    # since that word is not present anymore in the lattice.
    # With set +e, we are ignoring these failures.
    set +e;
    (
      lattice-push "ark:zcat $ilat|" ark:- |
      lattice-align-words-lexicon \
        decode/lm/original_lm/word_align_lexicon.txt \
        decode/lm/original_lm/model \
        ark:- "ark:|gzip -9 -c > $olat";
    ) &> "$latdir/aligned/lattice-align-words-lexicon.log";
    set -e;
    if [[ "$convert_to_slf" = true ]]; then
      lattice-copy "ark:zcat $olat|" ark,t:- |
      ./utils/int2sym.pl -f 3 decode/lm/original_lm/words2.txt |
      ./utils/convert_slf.pl --frame-rate 1 - "$latdir/aligned/slf";
    fi;
  done
done;



