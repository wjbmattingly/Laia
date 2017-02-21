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

eps="<eps>";
ctc="<ctc>";
sbeg="<s>";
send="</s>";
sil="<silence>";
unk="<unk>";
dummy="<dummy>";
wspace="<space>";
stroke="<stroke>";
transition_scale=1;
loop_scale=0.1;
overwrite=false;
help_message="
Usage: ${0##*/} [options] char_syms_in word_syms_in G_fst_in output_dir

Arguments:
  char_syms_in       : File mapping from character/HMM symbols to integer IDs.
                       This should be the file used by Laia during CTC training.
  word_syms_in       : File mapping from word symbols to integer IDs.
                       This should be the file used to create the word-LM FST.

Options:
  --ctc              : (type = string, default = \"$ctc\")
  --eps              : (type = string, default = \"$eps\")
  --dummy            : (type = string, default = \"$dummy\")
  --loop_scale       : (type = float, default = $loop_scale)
  --overwrite        : (type = boolean, default = $overwrite)
  --sbeg             : (type = string, default = \"$sbeg\")
  --send             : (type = string, default = \"$send\")
  --sil              : (type = string, default = \"$sil\")
  --stroke           : (type = string, default = \"$stroke\")
  --transition_scale : (type = float, default = $transition_scale)
  --unk              : (type = string, default = \"$unk\")
  --wspace           : (type = string, default = \"$wspace\")
";
source "$(pwd)/utils/parse_options.inc.sh" || exit 1;
[ $# -ne 4 ] && echo "$help_message" >&2 && exit 1;

char_syms_in="$1";
word_syms_in="$2";
G_fst_in="$3";
odir="$4";

mkdir -p "$odir";


# Create simplified lexicon with probabilities that will be used by
# the create_ctc_word_lexicon.sh script.
tmpf="$(mktemp)";
awk -v eps="$eps" -v sb="$sbeg" -v se="$send" -v sil="$sil" -v unk="$unk" \
  -v stroke="$stroke" 'BEGIN{
  IGNORE[eps]=1;
  IGNORE[sb]=1;
  IGNORE[se]=1;
  IGNORE[sil]=1;
  IGNORE[unk]=1;
  SPECIAL[stroke]=1;
}$1 !~ /\*$/ && !($1 in IGNORE){
  printf("%-25s 1.0", $1);
  if ($1 in SPECIAL) {
    printf(" %s\n", $1);
  } else {
    for (i=1;i<=length($1);++i) {
      c = substr($1, i, 1);
      c = (c == "#"  ? stroke : c);
      c = (c == "\"" ? "'\'' '\''" : c);
      printf(" %s", c);
    }
    printf("\n");
  }
}' "$word_syms_in" > "$tmpf";


# Create list of words that MUST NOT BE preceded by white spaces, since
# these were actually artifically separated to train the language model.
tmpf2="$(mktemp)"
cat <<EOF | awk -v WF="$word_syms_in" 'BEGIN{ while((getline < WF) > 0) W[$1]=1; }($1 in W)' > "$tmpf2"
'd
'll
'm
're
's
't
've
'D
'LL
'M
'RE
'S
'T
'VE
EOF


# Create the lexicon FST with disambiguation symbols.
./utils/create_ctc_word_lexicon.sh \
  --ctc "$ctc" --eps "$eps" --dummy "$dummy" --wspace "$wspace" \
  --no_space_words "$tmpf2" --overwrite "$overwrite" \
  "$char_syms_in" "$word_syms_in" "$tmpf" "$odir/chars.txt" "$odir/words.txt" \
  "$odir/L.fst" || ( echo "Failed $odir/L.fst creation!" && exit 1; );


# Create integer lists of disambiguation symbols.
awk '$1 ~ /^#.+/{ print $2 }' "$odir/chars.txt" > "$odir/chars_disambig.int";
awk '$1 ~ /^#.+/{ print $2 }' "$odir/words.txt" > "$odir/words_disambig.int";


# Compose the context-dependent and the L transducers.
[[ "$overwrite" = false && -s "$odir/CL.fst" &&
    ( ! "$odir/CL.fst" -ot "$odir/L.fst" ) ]] ||
fstcomposecontext --context-size=1 --central-position=0 \
  --read-disambig-syms="$odir/chars_disambig.int" \
  --write-disambig-syms="$odir/ilabels_disambig.int" \
  "$odir/ilabels" "$odir/L.fst" |
fstarcsort --sort_type=ilabel > "$odir/CL.fst" ||
( echo "Failed $odir/CL.fst creation!" && exit 1; );


# Create HMM model and tree
./utils/create_ctc_hmm_model.sh --eps "$eps" --ctc "$ctc" --dummy "$dummy" \
  --overwrite "$overwrite" "$odir/chars.txt" "$odir/model" "$odir/tree";


# Create Ha transducer
[[ "$overwrite" = false && -s "$odir/Ha.fst" &&
    ( ! "$odir/Ha.fst" -ot "$odir/model" ) &&
    ( ! "$odir/Ha.fst" -ot "$odir/tree" ) &&
    ( ! "$odir/Ha.fst" -ot "$odir/ilabels" ) ]] ||
make-h-transducer --disambig-syms-out="$odir/tid_disambig.int" \
  --transition-scale="$transition_scale" "$odir/ilabels" "$odir/tree" \
  "$odir/model" > "$odir/Ha.fst" ||
( echo "Failed $odir/Ha.fst creation!" && exit 1; );


# Create HaCL transducer.
[[ "$overwrite" = false && -s "$odir/HCL.fst" &&
    ( ! "$odir/HaCL.fst" -ot "$odir/Ha.fst" ) &&
    ( ! "$odir/HCL.fst" -ot "$odir/CL.fst" ) ]] ||
fsttablecompose "$odir/Ha.fst" "$odir/CL.fst" |
fstdeterminizestar --use-log=true |
fstrmsymbols "$odir/tid_disambig.int" |
fstrmepslocal |
fstminimizeencoded > "$odir/HaCL.fst" ||
( echo "Failed $odir/HaCL.fst creation!" && exit 1; );


# Create HCL transducer.
[[ "$overwrite" = false && -s "$odir/HCL.fst" &&
    ( ! "$odir/HCL.fst" -ot "$odir/HaCL.fst" ) ]] ||
add-self-loops --self-loop-scale="$loop_scale" --reorder=true \
  "$odir/model" "$odir/HaCL.fst" |
fstarcsort --sort_type=olabel > "$odir/HCL.fst" ||
( echo "Failed $odir/HCL.fst creation!" && exit 1; );


# Create the grammar FST from the ARPA language model.
[[ "$overwrite" = false && -s "$odir/G.fst" &&
    ( ! "$odir/G.fst" -ot "$G_fst_in" ) &&
    ( ! "$odir/G.fst" -ot "$odir/chars.txt" ) &&
    ( ! "$odir/G.fst" -ot "$odir/words.txt" ) ]] ||
fstprint --isymbols="$word_syms_in" --osymbols="$word_syms_in" "$G_fst_in" |
awk -v dm="$dummy" -v eps="$eps" -v sil="<sil>" -v unk="<unk>" \
  -v sbeg="$sbeg" -v send="$send" \
  -v stroke="$stroke" '
{
  if (NF >= 4) {
    # Replace <eps> by #0 in the input (this is like remove_oovs.pl)
    if ($3==eps) $3="#0";
    # Remove <silence>, Aachen mapped it to epsilon in the lexicon
    if ($3==sil || $4==sil) next;
    # Ignore arcs with unknown, since they are not mapped in the lexicon.
    if ($3==unk || $4==unk) next;
    # Remove <s> from the input and output
    if ($3==sbeg) $3=eps; if ($4==sbeg) $4=eps;
    # Remove </s> from the output, replace by <dummy> in the input
    if ($3==send) $3=dm; if ($4==send) $4=eps;
  }
  print
}' |
fstcompile --isymbols="$odir/words.txt" --osymbols="$odir/words.txt" |
fstconnect |
fstdeterminizestar --use-log=true |
fstarcsort --sort_type=ilabel > "$odir/G.fst" ||
( echo "Failed $odir/G.fst creation!" && exit 1; );


exit 0;
