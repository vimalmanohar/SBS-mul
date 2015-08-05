#!/bin/bash -e

. ./cmd.sh
. ./path.sh

decode_nj=4

SBS_LANG="MD SW AR DT HG UR"

gmmdir=exp/tri3b
data_fmllr=data-fmllr-tri3b

. utils/parse_options.sh

# Decode (reuse HCLG graph)
for lang in $SBS_LANG; do
  dir=exp/dnn4_pretrain-dbn_dnn

  steps/nnet/decode.sh --nj $decode_nj --cmd "$decode_cmd" --acwt 0.2 \
    $gmmdir/graph $data_fmllr/dev_$lang $dir/decode_dev_$lang &
done
wait

echo Success
exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
