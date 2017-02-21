#!/bin/bash
set -e;
export LC_NUMERIC=C;

with_id=0;
[ "$1" = "--with-id" ] && with_id=1 && shift 1;

awk -v with_id=$with_id '{
 t="";
 if (with_id) { t=$1" "; $1=""; $0=substr($0, 2); };
 print t""gensub(/ '\''(s|d|ll|m|ve|t|re|S|D|LL|M|VE|T|RE)\y/, "'\''\\1", "g");
}' $@;

exit 0;