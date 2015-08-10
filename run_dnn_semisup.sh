#!/bin/bash -e

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

set -e 
set -o pipefail
set -u

# Set the location of the SBS speech
SBS_CORPUS=/export/ws15-pt-data/data/audio

LANG="SW"   # Target language

feats_nj=40
train_nj=20
decode_nj=5
stage=-100 # resume training with --stage=N

# Semi-supervised training options
num_copies=3    # Make this many copies of supervised data
threshold=      # If provided, use frame thresholding -- keep only frames whose
                # best path posterior is above this value
use_soft_counts=false

# Decode Config:
acwt=0.2
parallel_opts="--num-threads 6"

gmmdir=exp/tri3b
data_fmllr=data-fmllr-tri3b
graph_dir=exp/tri3b/graph
feature_transform=exp/dnn4_pretrain-dbn/final.feature_transform
dbn=exp/dnn4_pretrain-dbn/6.dbn
dnndir=exp/dnn4_pretrain-dbn_dnn

dir=exp/dnn5_pretrain-dbn_dnn_semisup

# End of config.

set -o pipefail
set -e
set -u 

. utils/parse_options.sh

L=$LANG

if [ $stage -le -4 ]; then
  local/sbs_gen_data_dir.sh --corpus-dir=$SBS_CORPUS \
    --lang-map=conf/lang_codes.txt $LANG || exit 1
fi

if [ $stage -le -3 ]; then
  mfccdir=mfcc/$L
  steps/make_mfcc.sh --nj $feats_nj --cmd "$train_cmd" data/$L/unsup exp/$L/make_mfcc/unsup $mfccdir || exit 1

  utils/subset_data_dir.sh data/$L/unsup 4000 data/$L/unsup_4k || exit 1
  steps/compute_cmvn_stats.sh data/$L/unsup_4k exp/$L/make_mfcc/unsup_4k $mfccdir || exit 1
fi

graph_affix=${graph_dir#*graph}

if [ $stage -le -2 ]; then
  steps/decode_fmllr.sh $parallel_opts --nj $train_nj --cmd "$decode_cmd" \
    --skip-scoring true --acwt $acwt \
    $graph_dir data/$L/unsup_4k $gmmdir/decode${graph_affix}_unsup_4k_$L || exit 1
fi

if [ $stage -le -1 ]; then
  featdir=$data_fmllr/unsup_4k_$L
  steps/nnet/make_fmllr_feats.sh --nj $feats_nj --cmd "$train_cmd" \
    --transform-dir $gmmdir/decode${graph_affix}_unsup_4k_$L \
    $featdir data/$L/unsup_4k $gmmdir $featdir/log $featdir/data  || exit 1
fi

decode_dir=$dnndir/decode${graph_affix}_unsup_4k_$L
best_path_dir=$dnndir/best_path${graph_affix}_unsup_4k_$L

if [ $stage -le 0 ]; then
  steps/nnet/decode.sh $parallel_opts --nj $train_nj --cmd "$decode_cmd" \
    --acwt $acwt --skip-scoring true \
    $graph_dir $data_fmllr/unsup_4k_$L $decode_dir || exit 1
fi


postdir=$dnndir/post${graph_affix}_semisup_4k${threshold:-_$threshold}

if [ $stage -le 1 ]; then
  L=$LANG
  local/best_path_weights.sh --acwt $acwt data/$L/unsup_4k $graph_dir \
    $decode_dir $dnndir/best_path${graph_affix}_unsup_4k_$L || exit 1
fi


if [ $stage -le 2 ]; then
  nj=$(cat $gmmdir/num_jobs)
  $train_cmd JOB=1:$nj $postdir/get_train_post.JOB.log \
    ali-to-pdf $gmmdir/final.mdl "ark:gunzip -c $gmmdir/ali.JOB.gz |" ark:- \| \
    ali-to-post ark:- ark,scp:$postdir/train_post.JOB.ark,$postdir/train_post.JOB.scp || exit 1
  for n in `seq $nj`; do 
    cat $postdir/train_post.$n.scp
  done > $postdir/train_post.scp

  for n in `seq $nj`; do
    copy-int-vector "ark:gunzip -c $gmmdir/ali.$n.gz |" ark,t:- 
  done | \
    awk '{printf $1" ["; for (i=2; i<=NF; i++) { printf " "1; }; print " ]";}' | \
    copy-vector ark,t:- ark,scp:$postdir/train_frame_weights.ark,$postdir/train_frame_weights.scp || exit 1
fi

if [ $stage -le 3 ]; then
  nj=$(cat $best_path_dir/num_jobs)
  if ! $use_soft_counts; then
    $train_cmd JOB=1:$nj $postdir/get_unsup_post.JOB.log \
      ali-to-pdf $gmmdir/final.mdl "ark:gunzip -c $best_path_dir/ali.JOB.gz |" ark:- \| \
      ali-to-post ark:- ark,scp:$postdir/unsup_post.JOB.ark,$postdir/unsup_post.JOB.scp || exit 1
  else 
    $train_cmd JOB=1:$nj $postdir/get_unsup_soft_post.JOB.log \
      lattice-to-post --acoustic-scale=$acwt "ark:gunzip -c $decode_dir/lat.JOB.gz |" ark:- \| \
      post-to-pdf-post $gmmdir/final.mdl ark:- \
      ark,scp:$postdir/unsup_post.JOB.ark,$postdir/unsup_post.JOB.scp || exit 1
  fi
  for n in `seq $nj`; do
    cat $postdir/unsup_post.$n.scp
  done > $postdir/unsup_post.scp

  copy_command=copy-vector
  if [ ! -z "$threshold" ]; then
    copy_command="thresh-vector --threshold=$threshold --lower-cap=0.0 --upper-cap=1.0"
  fi

  $train_cmd JOB=1:$nj $postdir/copy_frame_weights.JOB.log \
    $copy_command "ark:gunzip -c $best_path_dir/weights.JOB.gz |" \
    ark,scp:$postdir/unsup_frame_weights.JOB.ark,$postdir/unsup_frame_weights.JOB.scp || exit 1
  
  for n in `seq $nj`; do
    cat $postdir/unsup_frame_weights.$n.scp
  done > $postdir/unsup_frame_weights.scp
fi

if [ $stage -le 4 ]; then
  awk -v num_copies=$num_copies \
    '{for (i=0; i<num_copies; i++) { print i"-"$1" "$2 } }' \
    $postdir/train_post.scp > $postdir/train_post_${num_copies}x.scp
  
  awk -v num_copies=$num_copies \
    '{for (i=0; i<num_copies; i++) { print i"-"$1" "$2 } }' \
    $postdir/train_frame_weights.scp > $postdir/train_frame_weights_${num_copies}x.scp

  copied_data_dirs=
  for i in `seq 0 $[num_copies-1]`; do
    utils/copy_data_dir.sh --utt-prefix ${i}- --spk-prefix ${i}- $data_fmllr/train_tr90 \
      $data_fmllr/train_tr90_$i || exit 1
    copied_data_dirs="$copied_data_dirs $data_fmllr/train_tr90_$i"
  done

  utils/combine_data.sh $data_fmllr/train_tr90_${num_copies}x $copied_data_dirs || exit 1
fi

if [ $stage -le 5 ]; then
  utils/combine_data.sh $dir/data_semisup_4k_${num_copies}x $data_fmllr/unsup_4k_$L $data_fmllr/train_tr90_${num_copies}x  || exit 1
  utils/copy_data_dir.sh --utt-prefix 0- --spk-prefix 0- $data_fmllr/train_cv10 \
    $data_fmllr/train_cv10_0 || exit 1
  
  sort -k1,1 $postdir/unsup_post.scp $postdir/train_post_${num_copies}x.scp > $dir/all_post.scp
  sort -k1,1 $postdir/unsup_frame_weights.scp $postdir/train_frame_weights_${num_copies}x.scp > $dir/all_frame_weights.scp

  num_tgt=$(hmm-info --print-args=false $gmmdir/final.mdl | grep pdfs | awk '{ print $NF }')
  $cuda_cmd $dir/log/train.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn \
    --hid-layers 0 --learn-rate 0.008 --num-tgt $num_tgt \
    --labels scp:$dir/all_post.scp --frame-weights scp:$dir/all_frame_weights.scp \
    $dir/data_semisup_4k_${num_copies}x $data_fmllr/train_cv10_0 \
    data/$L/lang dummy dummy $dir || exit 1;
fi

if [ $stage -le 6 ]; then
  steps/nnet/make_priors.sh --cmd "$train_cmd" --nj $train_nj $data_fmllr/train $dir || exit 1
  cp $dnndir/final.mdl $dir
fi

if [ $stage -le 7 ]; then
  # Decode (reuse HCLG graph)
  for lang in $L; do
    steps/nnet/decode.sh $parallel_opts --nj $decode_nj --cmd "$decode_cmd" --acwt $acwt \
      $graph_dir $data_fmllr/dev_$lang $dir/decode${graph_affix}_dev_$lang || exit 1
  done
fi
