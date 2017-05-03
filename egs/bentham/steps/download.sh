#!/bin/bash
set -e;
export LC_NUMERIC=C;

# Directory where the download.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/steps" != "$SDIR" ] && \
    echo "Please, run this script from the experiment top directory!" >&2 && \
    exit 1;

[[ -d data ]] || mkdir -p data;

# Download Bentham data
HTRtS_url="http://transcriptorium.eu/~htrcontest/contestICFHR2014/public_html";
data_url="$HTRtS_url/HTRtS2014/contestHTRtS.tbz";
data2_url="$HTRtS_url/HTRtS2014/contestHTRtS-test-with-Transcripts.tbz";
# Training + Validation
[[ -s data/original/contestHTRtS.tbz || -d data/original/contestHTRtS ]] ||
wget -P data/original "$data_url" ||
( echo "ERROR: Downloading data from $data_url" >&2 && exit 1 );
# Test (with transcripts)
[[ -s data/original/contestHTRtS-test-with-Transcripts.tbz ||
   -d data/original/contestHTRtS-test ]] ||
wget -P data/original "$data2_url" ||
( echo "ERROR: Downloading data from $data2_url" >&2 && exit 1 );

# Untar data
[[ -d data/original/contestHTRtS ]] ||
( tar xjf data/original/contestHTRtS.tbz -C data/original &&
  rm data/original/contestHTRtS.tbz ) ||
( echo "ERROR: Untarring data/original/contestHTRtS.tbz" >&2 && exit 1 );
[[ -d data/original/contestHTRtS-test ]] ||
( tar xjf data/original/contestHTRtS-test-with-Transcripts.tbz -C data &&
  rm data/original/contestHTRtS-test-with-Transcripts.tbz ) ||
( echo "ERROR: Untarring data/original/contestHTRtS-test-with-Transcripts.tbz" \
  >&2 && exit 1 );
