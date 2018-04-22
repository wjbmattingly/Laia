#!/bin/bash
set -e;

# Directory where this script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/steps" != "$SDIR" ] && \
    echo "Please, run this script from the experiment top directory!" >&2 && \
    exit 1;
[ ! -f "$(pwd)/utils/parse_options.inc.sh" ] && \
    echo "Missing $(pwd)/utils/parse_options.inc.sh file!" >&2 && exit 1;

overwrite=false;
help_message="
Usage: ${0##*/} [options]

Options:
  --overwrite  : (type = boolean, default = $overwrite)
                 Overwrite previously created files.
";
source "$(pwd)/utils/parse_options.inc.sh" || exit 1;

## Download lines images from FKI.
[ "$overwrite" = false -a -s data/washingtondb-v1.0.zip ] || {
  [ -z "$FKI_USER" -o -z "$FKI_PASSWORD" ] &&
  echo "Please, set the FKI_USER and FKI_PASSWORD variables to download the" \
       "Washington database from the FKI servers." >&2 && exit 1;
  wget -P data --user="$FKI_USER" --password="$FKI_PASSWORD" \
       http://www.fki.inf.unibe.ch/DBs/iamHistDB/data/washingtondb-v1.0.zip;
}

## Unzip dataset.
[ "$overwrite" = false -a -d data/washingtondb-v1.0 ] ||
( cd data && unzip -uqq washingtondb-v1.0.zip && \
  rm -rf __MACOSX && cd - &> /dev/null ) ||
exit 1;
