#!/bin/bash
set -e;

# Directory where the prepare.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/steps" != "$SDIR" ] && \
    echo "Please, run this script from the experiment top directory!" >&2 && \
    exit 1;
[ ! -f "$(pwd)/utils/parse_options.inc.sh" ] && \
    echo "Missing $(pwd)/utils/parse_options.inc.sh file!" >&2 && exit 1;

# Utility function to download files from FKI.
function download_url () {
  [ -z "$FKI_USER" -o -z "$FKI_PASSWORD" ] && \
    echo "Please, set the FKI_USER and FKI_PASSWORD variables to download the" \
    "IAM database from the FKI servers." >&2 && return 1;
  wget -P data/original --user="$FKI_USER" --password="$FKI_PASSWORD" "$1" ||
  { echo "ERROR: Failed downloading $1!" >&2 && return 1; }
  return 0;
}

mkdir -p data/original/lines;
tmpd="$(mktemp -d)";

# Download lines images from FKI.
[ -s data/original/lines.tgz ] ||
download_url http://www.fki.inf.unibe.ch/DBs/iamDB/data/lines/lines.tgz ||
exit 1;

# Untar and put all images into a single directory.
[ -s data/original/lines/a01-132x-08.png ] || (
  tar zxf data/original/lines.tgz -C "$tmpd" &&
  find "$tmpd" -name "*.png" | xargs -I{} mv {} data/original/lines; ) ||
( echo "ERROR: Failed extracting IAM line image files." >&2 && exit 1 );

# Download ascii files.
[ -s data/original/ascii.tgz ] ||
download_url http://www.fki.inf.unibe.ch/DBs/iamDB/data/ascii/ascii.tgz ||
exit 1;

# Untar ascii files.
[ -s data/original/lines.txt -a -s data/original/sentences.txt \
  -a -s data/original/forms.txt -a -s data/original/words.txt ] ||
tar zxf data/original/ascii.tgz -C data/original ||
( echo "ERROR: Failed extracting IAM ascii files." >&2 && exit 1 );

rm -rf "$tmpd";
