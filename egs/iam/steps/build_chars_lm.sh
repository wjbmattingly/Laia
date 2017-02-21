#!/bin/bash
export LC_NUMERIC=C;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/steps" = "$SDIR" ] || {
    echo "Run the script from \"$(dirname $SDIR)\"!" >&2 && exit 1;
}

order=10;
overwrite=false;
srilm_options="-wbdiscount -interpolate";
help_message="
Usage: ${0##*/} [--order N] [--overwrite true|false] data_dir lm_dir

Options:
  --order           : (int, value = $order)
                      Maximum order of N-grams to count.
  --overwrite       : (int, value = $overwrite)
                      Overwrite existing files from previous runs.
  --srilm_options   : (string, value = \"$srilm_options\")
                      Options passed to SRILM's ngram-count.
";
. utils/parse_options.inc.sh || exit 1;

[ $# -ne 2 ] && echo "$help_message" >&2 && exit 1;
ddir="$1";
odir="$2";

mkdir -p "$odir/char";

char_lst="$(mktemp)";
awk '{print $1}' $ddir/lang/char/symbs.txt | tail -n+3 > "$char_lst";

va_txt="$(mktemp)";
cut -d\  -f2- $ddir/lang/char/va.txt > "$va_txt";
tr_txt="$(mktemp)";
cut -d\  -f2- $ddir/lang/char/tr.txt > "$tr_txt";

# Build n-grams.
for c in brown wellington lob_excludealltestsets; do
  # Create character-level n-gram from the external data. I use Witten-Bell
  # discount instead of the traditional Kneser-Ney because of the lack of
  # unigram singletons.
  [ -s "$odir/char/${c}-${order}gram.arpa.gz" -a "${overwrite}" = false ] ||
  ngram-count -order ${order} ${srilm_options} -vocab "$char_lst" \
    -text "$ddir/lang/char/$c.txt" -lm - |
  gzip -9 -c > "$odir/char/${c}-${order}gram.arpa.gz";

  # Compute detailed perplexity on the character-level validation set.
  [ -s "$odir/char/${c}-${order}gram.info" -a "${overwrite}" = false ] ||
  zcat "$odir/char/${c}-${order}gram.arpa.gz" |
  ngram -order ${order} -vocab "$char_lst" -ppl "$va_txt" -lm - -debug 2 \
    > "$odir/char/${c}-${order}gram.info";
done;

# Create char-level n-gram from the train partition
[ -s "$odir/char/train-${order}gram.arpa.gz" -a "${overwrite}" = false ] ||
ngram-count -order ${order} ${srilm_options} -vocab "$char_lst" \
  -text "$tr_txt" -lm - | gzip -9 -c > "$odir/char/train-${order}gram.arpa.gz";

# Compute detailed perplexity on the character-level validation set.
[ -s "$odir/char/train-${order}gram.info" -a "${overwrite}" = false ] ||
zcat "$odir/char/train-${order}gram.arpa.gz" |
ngram -order ${order} -vocab "$char_lst" -ppl "$va_txt" -lm - -debug 2 \
  > "$odir/char/train-${order}gram.info";

# Compute interpolation weights for the character-level n-gram
[ -s "$odir/char/interpolation-${order}gram.mix" -a "${overwrite}" = false ] ||
compute-best-mix \
  "$odir/char/train-${order}gram.info" \
  "$odir/char/brown-${order}gram.info" \
  "$odir/char/lob_excludealltestsets-${order}gram.info" \
  "$odir/char/wellington-${order}gram.info" \
  &> "$odir/char/interpolation-${order}gram.mix";

# Interpolate character-level n-grams
lambdas=( $(grep "best lambda" "$odir/char/interpolation-${order}gram.mix" | \
  awk -F\( '{print $2}' | tr -d \)) );
[ -s "$odir/char/interpolation-${order}gram.arpa.gz" \
  -a "${overwrite}" = false ] ||
ngram -order ${order} -vocab "$char_lst" \
  -lm <(zcat "$odir/char/train-${order}gram.arpa.gz") \
  -mix-lm <(zcat "$odir/char/brown-${order}gram.arpa.gz") \
  -mix-lm2 <(zcat "$odir/char/lob_excludealltestsets-${order}gram.arpa.gz") \
  -mix-lm3 <(zcat "$odir/char/wellington-${order}gram.arpa.gz") \
  -lambda ${lambdas[0]} \
  -mix-lambda2 ${lambdas[2]} \
  -mix-lambda3 ${lambdas[3]} \
  -write-lm - | gzip -9 > "$odir/char/interpolation-${order}gram.arpa.gz";

# Print char-level perplexity on the validation data set.
char_ppl=$(egrep -o "ppl = [0-9.]+$" \
  "$odir/char/interpolation-${order}gram.mix" | tail -n 1);
echo "Char-level ($order-gram) $char_ppl";

# Remove temporal files
rm -f "$char_lst" "$va_txt" "$tr_txt";

exit 0;
