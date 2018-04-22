#!/bin/bash
set -e;
export LC_NUMERIC=C;
export LUA_PATH="$(pwd)/../../?/init.lua;$(pwd)/../../?.lua;$LUA_PATH";

batch_size=16;
height=120;
overwrite=false;

## Directory where the run.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)" != "$SDIR" ] &&
echo "Please, run this script from the directory \"$SDIR\"!" >&2 &&
exit 1;

## Download data from FKI.
./steps/download.sh --overwrite "$overwrite";

## Prepare data for Laia training.
./steps/prepare.sh --height "$height" --overwrite "$overwrite";

## Create & train models.
./steps/train.sh --batch_size "$batch_size" --height "$height" --overwrite "$overwrite";

## Decode using the trained models.
./steps/decode_net.sh --overwrite "$overwrite";
