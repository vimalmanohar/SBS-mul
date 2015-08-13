#!/bin/bash

# Copyright 2015    Vimal Manohar
# Apache 2.0

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

set -e 
set -o pipefail
set -u

LANG=MD

srcdir=exp/dnn4_pretrain-dbn_dnn    # DNN to be adapted
gmmdir=exp/tri3b_ali_map_${LANG}_pt     # GMM dir to get transforms from
data_fmllr=data-fmllr-tri3b             # Feats matching those from gmmdir
postdir=exp/tri3b_map_${LANG}_pt        # Directory to get posteriors from

test_dir=data/${LANG}/dev           
test_transform_dir=exp/tri3b/decode_text_G_dev   # Test transform dir. Must match the GMM dir model.
graph_dir=exp/tri3b_map_${LANG}_pt//graph_text_G  # Graph from the posterior directory
stage=0 # resume training with --stage=N

prune_threshold=0.7
threshold=0.7
acwt=0.2
other_lr=1

feats_nj=10
train_nj=10
decode_nj=5

reinit_softmax=true

dir=${srcdir}_${LANG}_pt_${prune_threshold}_ot${other_lr}

. utils/parse_options.sh 

if [ $stage -le 0 ]; then
  data_dir=$data_fmllr/$LANG/train
  steps/nnet/make_fmllr_feats.sh --nj $feats_nj --cmd "$train_cmd" \
    --transform-dir $gmmdir \
    $data_dir data/$LANG/train $gmmdir $data_dir/log $data_dir/data || exit 1
  utils/subset_data_dir_tr_cv.sh $data_dir ${data_dir}_tr90 ${data_dir}_cv10 || exit 1
fi

nj=$(cat $postdir/num_jobs) || exit 1
frame_weights_dir=${postdir}/frame_weights
if [ $stage -le 1 ]; then
  $train_cmd JOB=1:$nj $frame_weights_dir/log/get_frame_weights.JOB.log \
    copy-post --prune-threshold=$prune_threshold "ark:gunzip -c $postdir/post.JOB.gz | post-to-pdf-post $postdir/final.mdl ark:- ark:- |" ark:- \| \
    post-to-weights ark:- ark:- \| \
    thresh-vector --threshold=$threshold --lower-cap=0.0 ark:- \
    ark,scp:$frame_weights_dir/frame_weights.JOB.ark,$frame_weights_dir/frame_weights.JOB.scp || exit 1

  for n in `seq $nj`; do 
    cat $frame_weights_dir/frame_weights.$n.scp
  done > $frame_weights_dir/frame_weights.scp
fi

mkdir -p $dir/log

feature_transform=exp/dnn4_pretrain-dbn/final.feature_transform
if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  # Train
  if $reinit_softmax; then
    local/utils/nnet/renew_nnet_softmax.sh $postdir/final.mdl $srcdir/final.nnet $dir/prepre_init.nnet > $dir/log/prepre_init.log
  else 
    cp $srcdir/final.nnet $dir/prepre_init.nnet
  fi
  $train_cmd $dir/log/pre_init.log \
    nnet-copy --learning-rate-scales="$other_lr:$other_lr:$other_lr:$other_lr:$other_lr:$other_lr:1" $dir/prepre_init.nnet $dir/pre_init.nnet
  cp $postdir/final.mdl $dir    
  cp $gmmdir/final.mat $dir     # Feats match $gmmdir. Pdf match $postdir.
  cp $postdir/tree $dir

  idx_list="{`seq -s',' $nj`}"
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --nnet-init $dir/pre_init.nnet --hid-layers 0 --learn-rate 0.008 \
    --labels "ark:gunzip -c $postdir/post.$idx_list.gz | post-to-pdf-post $postdir/final.mdl ark:- ark:- |" \
    --frame-weights scp:$frame_weights_dir/frame_weights.scp \
    $data_fmllr/$LANG/train_tr90 $data_fmllr/$LANG/train_cv10 data/$LANG/lang dummy dummy $dir || exit 1;
fi

if [ $stage -le 3 ]; then
  steps/nnet/make_priors.sh --cmd "$train_cmd" --nj $train_nj $data_fmllr/$LANG/train $dir
fi

if [ $stage -le 4 ]; then
  test_id=`basename $test_dir`
  steps/nnet/make_fmllr_feats.sh --nj $decode_nj --cmd "$train_cmd" \
    --transform-dir $test_transform_dir $data_fmllr/${test_id}_$LANG \
    $test_dir $gmmdir exp/make_fmllr_feats/${test_id}_$LANG/log fmllr_feats/${test_id}_$LANG || exit 1
fi

if [ $stage -le 5 ]; then
  # Decode (reuse HCLG graph)
  graph_id=${graph_dir#*graph}
  test_id=`basename $test_dir`
  steps/nnet/decode.sh --nj $decode_nj --cmd "$decode_cmd" --acwt $acwt \
    $graph_dir $data_fmllr/${test_id}_$LANG $dir/decode${graph_id}_acwt${acwt}_${test_id}_$LANG || exit 1
fi
