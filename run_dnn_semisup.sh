#!/bin/bash -e

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

# Set the location of the SBS speech
SBS_CORPUS=/export/ws15-pt-data/data/audio

feats_nj=40
train_nj=10
decode_nj=5
parallel_opts="--num-threads 6"

LANG="SW"

# Config:
acwt=0.2
gmmdir=exp/tri3b
data_fmllr=data-fmllr-tri3b
dnndir=exp/dnn4_pretrain-dbn_dnn
stage=-100 # resume training with --stage=N
# End of config.

set -o pipefail
set -e
set -u 

. utils/parse_options.sh

L=$LANG

if [ $stage -le -4 ]; then
  local/sbs_gen_data_dir.sh --corpus-dir=$SBS_CORPUS \
    --lang-map=conf/lang_codes.txt $LANG
fi

if [ $stage -le -3 ]; then
  mfccdir=mfcc/$L
  steps/make_mfcc.sh --nj $feats_nj --cmd "$train_cmd" data/$L/unsup exp/$L/make_mfcc/unsup $mfccdir

  utils/subset_data_dir.sh 4000 data/$L/unsup data/$L/unsup_4k
  steps/compute_cmvn_stats.sh data/$L/unsup_4k exp/$L/make_mfcc/unsup_4k $mfccdir
fi

graph_dir=$gmmdir/graph
if [ $stage -le -2 ]; then
  steps/decode_fmllr.sh $parallel_opts --nj $train_nj --cmd "$decode_cmd" \
    $graph_dir data/$L/unsup_4k $gmmdir/decode_unsup_4k_$L
fi

if [ $stage -le -1 ]; then
  featdir=$data_fmllr/unsup_4k_$L
  steps/nnet/make_fmllr_feat.sh --nj $feats_nj --cmd "$train_cmd" \
    --transform-dir $gmmdir/decode_unsup_4k_$L \
    $featdir data/$L/unsup_4k $gmmdir $featdir/log $featdir/data 
fi

if [ $stage -le 0 ]; then
  steps/nnet/decode.sh --nj $train_nj --cmd "$decode_cmd" \
    --acwt $acwt \
    $graph_dir $data_fmllr/unsup_4k_$L $dnndir/decode_unsup_4k_$L
fi

if [ $stage -le 1 ]; then
  L=$LANG
  local/best_path_weights.sh data/$L/unsup_4k $graph_dir \
    $dnndir/decode_unsup_4k_$L $dnndir/best_path_unsup_4k_$L
fi

postdir=$dir/post_semisup_4k
if [ $stage -le 2 ]; then
  nj=$(cat $gmmdir/num_jobs)
  $train_cmd JOB=1:$nj $postdir/get_train_post.JOB.log \
    ali-to-pdf $gmmdir/final.mdl "ark:gunzip -c $gmmdir/ali.JOB.gz |" ark:- \| \
    ali-to-post ark:- ark,scp:$postdir/train_post.JOB.ark,$postdir/train_post.JOB.scp || exit 1
  for n in `seq $nj`; do 
    cat $postdir/train_post.$n.scp
  done > $postdir/train_post.scp

  for n in `seq $nj`; do
    gunzip -c $gmmdir/ali.$n.gz
  done | \
    awk '{printf $1" ["; for (i=2; i<=NF; i++) { printf " "1; }; print " ]";}' | \
    copy-vector ark,t:- ark,scp:$postdir/train_frame_weights.ark,$postdir/train_frame_weights.scp || exit 1
fi

if [ $stage -le 3 ]; then
  awk -v num_copies=$num_copies \
    '{for (i=0; i<num_copies; i++) { print i"-"$1" "$2 } }' \
    $postdir/train_post.scp > $postdir/train_post_${num_copies}x.scp
  
  awk -v num_copies=$num_copies \
    '{for (i=0; i<num_copies; i++) { print i"-"$1" "$2 } }' \
    $postdir/train_frame_weights.scp > $postdir/train_frame_weights_${num_copies}x.scp

  copied_data_dirs=
  for i in `seq 0 $[num_copies-1]`; do
    utils/copy_data_dir.sh --utt-prefix ${i}- $data_fmllr/train_tr90 \
      $data_fmllr/train_tr90_$i
    copied_data_dirs="$copied_data_dirs $data_fmllr/train_tr90_$i"
  done

  utils/combine_data_dir.sh $data_fmllr/train_90_${num_copies}x $copied_data_dirs
  rm -rf $copied_data_dirs
fi

if [ $stage -le 4 ]; then
  nj=$(cat $best_path_dir/num_jobs)
  $train_cmd JOB=1:$nj $postdir/get_unsup_post.JOB.log \
    ali-to-pdf $gmmdir/final.mdl "ark:gunzip -c $best_path_dir/ali.JOB.gz |" ark:- \| \
    ali-to-post ark:- ark,scp:$postdir/unsup_post.JOB.ark,$postdir/unsup_post.JOB.scp || exit 1

  for n in `seq $nj`; do
    cat $postdir/unsup_post.$n.scp
  done > $postdir/unsup_post.scp

  $train_cmd JOB=1:$nj $postdir/copy_frame_weights.JOB.log \
    copy-vector "ark:gunzip -c $best_path_dir/weights.JOB.gz |" \
    ark,scp:$postdir/unsup_frame_weights.JOB.ark,$postdir/unsup_frame_weights.JOB.scp || exit 1
  
  for n in `seq $nj`; do
    cat $postdir/unsup_frame_weights.$n.scp
  done > $postdir/unsup_frame_weights.scp
fi

feature_transform=exp/dnn4_pretrain-dbn/final.feature_transform
dbn=exp/dnn4_pretrain-dbn/6.dbn
dir=exp/dnn5_semisup

if [ $stage -le 5 ]; then
  utils/combine_data.sh $dir/data_semisup_4k_${num_copies}x $data_fmllr/$L/train_tr90_${num_copies}x $data_fmllr/$L/unsup_4k 
  
  sort -k1,1 $postdir/unsup_post.scp $postdir/train_post_${num_copies}x.scp > $dir/all_post.scp
  sort -k1,1 $postdir/unsup_frame_weights.scp $postdir/train_frame_weights.scp > $dir/all_frame_weights.scp

  steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn \
    --hid-layers 0 --learn-rate 0.008 \
    --labels scp:$dir/all_post.scp --frame-weights scp:all_frame_weights.scp \
    $dir/data_semisup_4k_${num_copies}x $data_fmllr/MD/train_cv10 \
    data/$L/lang dummy dummy $dir || exit 1;
fi
