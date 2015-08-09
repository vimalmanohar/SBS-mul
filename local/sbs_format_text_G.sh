#!/bin/bash -u

# Generating a phone bigram LM

set -o errexit
set -o pipefail

function read_dirname () {
  local dir_name=`expr "X$1" : '[^=]*=\(.*\)'`;
  [ -d "$dir_name" ] || { echo "Argument '$dir_name' not a directory" >&2; \
    exit 1; }
  local retval=`cd $dir_name 2>/dev/null && pwd || exit 1`
  echo $retval
}

PROG=`basename $0`;
usage="Usage: $PROG <2-letter language code>\n
Prepare phone bigram LM for an SBS language.\n\n
";

if [ $# -lt 1 ]; then
  echo -e $usage; exit 1;
fi

TEXT_PHONE_LM=/export/ws15-pt-data/data/text-phnlm

while [ $# -gt 0 ];
do
  case "$1" in
  --help) echo -e $usage; exit 0 ;;
  --text-phone-lm) TEXT_PHONE_LM=$2; shift; shift ;;
  ??) LCODE=$1; shift ;;
  *)  echo "Unknown argument: $1, exiting"; echo -e $usage; exit 1 ;;
  esac
done

[ -f path.sh ] && . path.sh  # Sets the PATH to contain necessary executables, incl. IRSTLM

echo "Preparing the language model G acceptor"


full_name=`awk '/'$LCODE'/ {print $2}' conf/lang_codes.txt`
test=data/${LCODE}/lang_test_text_G
mkdir -p $test
cp -r data/lang/* $test

fstprint --isymbols=data/lang_test/phones.txt --osymbols=data/lang_test/words.txt \
  data/lang_test/L.fst \
  | LC_ALL=en_US.UTF-8 local/project_phones.py data/${LCODE}/lang_test/phones.txt \
  | fstcompile --isymbols=data/lang_test/phones.txt --osymbols=data/lang_test/words.txt \
  > $test/L.fst

fstprint --isymbols=data/lang_test/phones.txt --osymbols=data/lang_test/words.txt \
  data/lang_test/L_disambig.fst \
  | LC_ALL=en_US.UTF-8 local/project_phones.py data/${LCODE}/lang_test/phones.txt \
  | fstcompile --isymbols=data/lang_test/phones.txt --osymbols=data/lang_test/words.txt \
  > $test/L_disambig.fst

cat $TEXT_PHONE_LM/${LCODE}/bigram.lm | \
  sed 's/\tg/\tɡ/g' | sed 's/ g/ ɡ/g' | \
  sed 's/sil/!SIL/g' | \
  egrep -v '<s> <s>|</s> <s>|</s> </s>' | \
  arpa2fst - | fstprint | \
  utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$test/words.txt \
    --osymbols=$test/words.txt  --keep_isymbols=false --keep_osymbols=false | \
  fstrmepsilon | fstarcsort --sort_type=ilabel > $test/G.fst
 
# fstisstochastic $test/G.fst
# The output is like:
# 9.14233e-05 -0.259833
# we do expect the first of these 2 numbers to be close to zero (the second is
# nonzero because the backoff weights make the states sum to >1).
# Because of the <s> fiasco for these particular LMs, the first number is not
# as close to zero as it could be.

local/validate_lang.pl $test || exit 1

echo "Done preparing G."
