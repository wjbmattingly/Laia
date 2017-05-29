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
beam=50;
lattice_beam=20;
max_active=5000000;
min_active=20;
num_procs="$(nproc)";
num_tasks="$(nproc)";
symbol_table=;
prune_interval=500;
tasks=;
overwrite=false;
qsub_opts="";
help_message="
Usage: ${0##*/} [options] loglikelihoods_scp fst_dir out_dir

Options:
  --acoustic_scale  : (float, default = $acoustic_scale)
                      Scaling factor for acoustic likelihoods.
  --beam            : (type = float, default = $beam)
                      Decoding beam.
  --lattice_beam    : (float, default = $lattice_beam)
                      Lattice generation beam.
  --max_active      : (type = integer, default = $max_active)
                      Max. number of tokens during Viterbi decoding
                      (a.k.a. histogram prunning).
  --min_active      : (type = integer, default = $min_active)
                      Decoder min active states (don't prune if #active < min).
  --num_procs       : (integer, default = $num_procs)
                      Maximum number of tasks to run in parallel in the host
                      computer (this maximum does not apply when using qsub).
  --num_tasks       : (integer, default = $num_tasks)
                      Divide the input scp in this number of independent tasks.
                      If --qsub_opts is given, these tasks will be executed in
                      parallel using SGE. If not, --num_procs processes will be
                      used in the local computer to process the tasks.
  --overwrite       : (type = boolean, default = $overwrite)
                      Overwrite ALL previous stages.
  --qsub_opts       : (type = string, default = \"$qsub_opts\")
                      If any option is given, will parallelize the decoding
                      using qsub. THIS IS HIGHLY RECOMMENDED.
  --prune_interval  : (integer, default = $prune_interval)
                      Interval (in frames) at which to prune tokens.
  --symbol_table    : (string, default = \"$symbol_table\")
                      Symbol table for for debug output.
  --tasks           : (string, default = \"$tasks\")
                      Range of tasks to execute. If not given, the range is set
                      automatically.
";
source utils/parse_options.inc.sh || exit 1;
[ $# -ne 3 ] && echo "$help_message" >&2 && exit 1;
lkh_scp="$1";
fst_dir="$2";
out_dir="$3";

# Check required files
for f in "$lkh_scp" "$fst_dir/model" "$fst_dir/HCL.fst" "$fst_dir/G.fst"; do
  [ ! -s "$f" ] && echo "ERROR: File \"$f\" was not found!" >&2 && exit 1;
done;

# Launch lattice generation.
./utils/latgen_lazy.sh \
  --acoustic_scale "$acoustic_scale" \
  --beam "$beam" \
  --lattice_beam "$lattice_beam" \
  --max_active "$max_active" \
  --min_active "$min_active" \
  --num_procs "$num_procs" \
  --num_tasks "$num_tasks" \
  --overwrite "$overwrite" \
  --qsub_opts "$qsub_opts" \
  --prune_interval "$prune_interval" \
  --symbol_table "$symbol_table" \
  --tasks "$tasks" \
  "$fst_dir/model" "$fst_dir/HCL.fst" "$fst_dir/G.fst" "$lkh_scp" "$out_dir";
