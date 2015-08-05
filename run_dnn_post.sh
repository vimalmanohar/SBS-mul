#!/bin/bash

# Copyright 2015    Vimal Manohar
# Apache 2.0

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

srcdir=exp/dnn4_pretrain-dbn_dnn
gmmdir=exp/tri3b_extra_MD_train_pt-2_2
data_fmllr=data-fmllr-tri3b
LANG=MD
postdir=exp/tri3b_extra_MD_train_pt-2_2
stage=0 # resume training with --stage=N
prune_threshold=0
feats_nj=10
train_nj=10
decode_nj=5

. utils/parse_options.sh 

if [ $stage -le 1 ]; then
  dir=$data_fmllr/$LANG/train
  steps/nnet/make_fmllr_feats.sh --nj $feats_nj --cmd "$train_cmd" \
    --transform-dir $gmmdir \
    $dir data/$LANG/train $gmmdir $dir/log $dir/data || exit 1
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
fi

dir=exp/dnn4_pretrain-dbn_dnn_${LANG}_pt_${prune_threshold}
feature_transform=exp/dnn4_pretrain-dbn/final.feature_transform
if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  # Train
  $train_cmd $dir/log/pre_init.log \
    nnet-copy --learning-rate-scales="0:0:0:0:0:0:1" $srcdir/final.nnet $dir/pre_init.nnet
  cp $gmmdir/final.mdl $dir
  cp $gmmdir/final.mat $dir
  cp $gmmdir/tree $dir
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --nnet-init $dir/pre_init.nnet --hid-layers 0 --learn-rate 0.008 \
    --labels "ark:gunzip -c $postdir/post.*.gz| post-to-pdf-post $postdir/final.mdl ark:- ark:- | copy-post --prune-threshold=$prune_threshold ark:- ark:- |" \
    $data_fmllr/MD/train_tr90 $data_fmllr/MD/train_cv10 data/$LANG/lang dummy dummy $dir || exit 1;
fi

if [ $stage -le 3 ]; then
  steps/nnet/make_priors.sh --cmd "$train_cmd" --nj $train_nj $data_fmllr/MD/train $dir
fi

if [ $stage -le 4 ]; then
  # Decode (reuse HCLG graph)
  for lang in $LANG; do
    steps/nnet/decode.sh --nj $decode_nj --cmd "$decode_cmd" --acwt 0.2 \
      $gmmdir/graph $data_fmllr/eval_$lang $dir/decode_eval_$lang || exit 1
  done
fi
