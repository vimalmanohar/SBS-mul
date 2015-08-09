#!/bin/bash -u

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
usage="Usage: $PROG <arguments> <2-letter language code>\n
Prepare unlabeled data directory for an SBS language.\n\n
Required arguments:\n
  --corpus-dir=DIR\tDirectory for the SBS (matched) corpus\n
  --lang-map=FILE\tMapping from 2-letter language code to full name\n
  ";

if [ $# -ne 3 ]; then
  echo -e $usage; exit 1;
fi

while [ $# -gt 0 ];
do
  case "$1" in
  --help) echo -e $usage; exit 0 ;;
  --corpus-dir=*) 
  SBSDIR=`read_dirname $1`; shift ;;
  --lang-map=*)
  LANGMAP=`expr "X$1" : '[^=]*=\(.*\)'`; shift ;;
  ??) LCODE=$1; shift ;;
  *)  echo "Unknown argument: $1, exiting"; echo -e $usage; exit 1 ;;
  esac
done

[ -f path.sh ] && . path.sh  # Sets the PATH to contain necessary executables

full_name=`awk '/'$LCODE'/ {print $2}' $LANGMAP`;

# Checking if sox is installed
which sox > /dev/null

mkdir -p data/$LCODE/wav/unsup # directory storing all the downsampled WAV files
tmpdir=$(mktemp -d);
echo $tmpdir
trap 'rm -rf "$tmpdir"' EXIT
mkdir -p $tmpdir
mkdir -p $tmpdir/downsample
mkdir -p $tmpdir/trans

soxerr=$tmpdir/soxerr;

find $SBSDIR/${full_name}_unlabeled -name "*.wav" > $tmpdir/wav_list
for x in `cat $tmpdir/wav_list`; do
  y=`basename $x`
  z=${y%*.wav}
  echo "$z $x" 
done > $tmpdir/wav_tmp_scp

cat data/$LCODE/train/wav.scp > $tmpdir/exclude_list
cat data/$LCODE/eval/wav.scp >> $tmpdir/exclude_list 
cat data/$LCODE/dev/wav.scp >> $tmpdir/exclude_list || true

utils/filter_scp.pl --exclude <(sort -k1,1 $tmpdir/exclude_list) $tmpdir/wav_tmp_scp | awk '{print $2}' > $tmpdir/unsup_wav_list

for x in `cat $tmpdir/unsup_wav_list`; do
  y=`basename $x`
  base=${y%*.wav}
  wavfile=$x
  outwavfile=data/$LCODE/wav/unsup/$base.wav

  [[ -e $outwavfile ]] || sox $wavfile -r 8000 -t wav $outwavfile 

  if [ $? -ne 0 ]; then
    echo "$wavfile: exit status = $?" >> $soxerr
    let "nsoxerr+=1"
  else 
    nsamples=`soxi -s "$outwavfile"`;
    if [[ "$nsamples" -gt 1000 ]]; then
      echo "$outwavfile" >> $tmpdir/downsample/unsup_wav
    else
      echo "$outwavfile: #samples = $nsamples" >> $soxerr;
      let "nsoxerr+=1"
    fi
  fi
done

sed -e "s:.*/::" -e 's:.wav$::' $tmpdir/downsample/unsup_wav > $tmpdir/downsample/unsup_basenames_wav

paste $tmpdir/downsample/unsup_basenames_wav $tmpdir/downsample/unsup_wav | sort -k1,1 > data/${LCODE}/local/data/unsup_wav.scp 

sed -e 's:\-.*$::' $tmpdir/downsample/unsup_basenames_wav | \
  paste -d' ' $tmpdir/downsample/unsup_basenames_wav - | sort -t' ' -k1,1 \
  > data/${LCODE}/local/data/unsup_utt2spk

./utils/utt2spk_to_spk2utt.pl data/${LCODE}/local/data/unsup_utt2spk > data/${LCODE}/local/data/unsup_spk2utt || exit 1;

mkdir -p data/$LCODE/unsup

cp data/$LCODE/local/data/unsup_wav.scp data/$LCODE/unsup/wav.scp
cp data/$LCODE/local/data/unsup_utt2spk data/$LCODE/unsup/utt2spk
cp data/$LCODE/local/data/unsup_spk2utt data/$LCODE/unsup/spk2utt

utils/fix_data_dir.sh data/$LCODE/unsup
