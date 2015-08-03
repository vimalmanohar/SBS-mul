#!/bin/bash

# this is our online-nnet2 build.  it's a "multi-splice" system (i.e. we have
# splicing at various layers), with p-norm nonlinearities.  We use the "accel2"
# script which uses between 2 and 14 GPUs depending how far through training it
# is.  You can safely reduce the --num-jobs-final to however many GPUs you have
# on your system.

# For joint training with RM, this script is run using the following command line,
# and note that the --stage 8 option is only needed in case you already ran the
# earlier stages.
# local/online/run_nnet2.sh --stage 8 --dir exp/nnet2_online/nnet_ms_a_partial --exit-train-stage 15

. cmd.sh

set -e
set -o pipefail
set -u

stage=0
train_stage=-10
use_gpu=true
dir=exp/nnet2_online/nnet_ms_a
ali_dir=exp/tri3b
train_data_dir=data/train
lang=data/lang
ivector_dir=exp/nnet2_online/ivectors_train

exit_train_stage=-100
. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.  Otherwise, call this script with --use-gpu false
EOF
  fi
  parallel_opts="--gpu 1" 
  num_threads=1
  minibatch_size=512
  # the _a is in case I want to change the parameters.
else
  num_threads=16
  minibatch_size=128
  parallel_opts="-pe smp $num_threads" 
fi

local/online/run_nnet2_common.sh --stage $stage || exit 1;

if [ $stage -le 8 ]; then
  # last splicing was instead: layer3/-4:2" 
  steps/nnet2/train_multisplice_accel2.sh --stage $train_stage \
    --exit-stage $exit_train_stage \
    --num-epochs 8 --num-jobs-initial 2 --num-jobs-final 14 \
    --num-hidden-layers 4 \
    --splice-indexes "layer0/-1:0:1 layer1/-2:1 layer2/-4:2" \
    --feat-type raw \
    --online-ivector-dir $ivector_dir \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --io-opts "--max-jobs-run 12" \
    --initial-effective-lrate 0.005 --final-effective-lrate 0.0005 \
    --cmd "$decode_cmd" \
    --pnorm-input-dim 1500 \
    --pnorm-output-dim 250 \
    ${train_data_dir}_hires $lang $ali_dir $dir  || exit 1;
fi

if [ $stage -le 9 ]; then
  # If this setup used PLP features, we'd have to give the option --feature-type plp
  # to the script below.
  iter_opt=
  [ $exit_train_stage -gt 0 ] && iter_opt="--iter $exit_train_stage"
  steps/online/nnet2/prepare_online_decoding.sh $iter_opt --mfcc-config conf/mfcc_hires.conf \
    $lang exp/nnet2_online/extractor "$dir" ${dir}_online || exit 1;
fi

if [ $exit_train_stage -gt 0 ]; then
  echo "$0: not testing since you only ran partial training (presumably in preparation"
  echo " for multilingual training"
  exit 0;
fi
