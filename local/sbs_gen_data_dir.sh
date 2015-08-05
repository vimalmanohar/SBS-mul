#!/bin/bash

if [ $# -ne 2 ]; then
  echo "sbs_gen_data_dir.sh <DATA> <data-dir>"
  exit 1
fi

DATA=$1
dir=$2

mkdir -p data/local/$dir
mkdir -p $dir

find $DATA -name "*.wav" > data/local/$dir/wav.scp
for x in `cat data/local/$dir/wav.scp`; do
  y=`basename $x`
  z=${y%*.wav}
  echo $z $x
done > $dir/wav.scp

awk '{print $1" "$1}' $dir/wav.scp > $dir/utt2spk
awk '{print $1" "$1}' $dir/wav.scp > $dir/spk2utt

utils/fix_data_dir.sh $dir
