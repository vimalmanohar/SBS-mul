#!/bin/bash

#
# This is supposed to be run after run.sh.
#

set -e

. ./cmd.sh
. ./path.sh

SBS_LANGUAGES="AR CA DT HG MD SW UR"

for L in $SBS_LANGUAGES; do
  echo "Prep oracle G for $L"
  local/sbs_format_oracle_LG.sh $L >& data/$L/format_oracle_LG.log
done

## Decode with oracle G
#for L in $SBS_LANGUAGES; do
#  graph_dir=exp/mono/$L/graph_oracle_LG
#  mkdir -p $graph_dir
#  utils/mkgraph.sh --mono data/$L/lang_test_oracle_LG exp/mono \
#    $graph_dir >& $graph_dir/mkgraph.log
#
#  steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
#    exp/mono/decode_eval_oracle_LG_$L &
#done
#wait
#
## Decode with oracle G
#for L in $SBS_LANGUAGES; do
#  graph_dir=exp/tri1/$L/graph_oracle_LG
#  mkdir -p $graph_dir
#  utils/mkgraph.sh data/$L/lang_test_oracle_LG exp/tri1 \
#    $graph_dir >& $graph_dir/mkgraph.log
#
#  steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
#    exp/tri1/decode_eval_oracle_LG_$L &
#done
#wait
#
## Decode with oracle G
#for L in $SBS_LANGUAGES; do
#  graph_dir=exp/tri2a/$L/graph_oracle_LG
#  mkdir -p $graph_dir
#  utils/mkgraph.sh data/$L/lang_test_oracle_LG exp/tri2a \
#    $graph_dir >& $graph_dir/mkgraph.log
#
#  steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
#    exp/tri2a/decode_eval_oracle_LG_$L &
#done
#wait
#
## Decode with oracle G
#for L in $SBS_LANGUAGES; do
#  graph_dir=exp/tri2b/$L/graph_oracle_LG
#  mkdir -p $graph_dir
#  utils/mkgraph.sh data/$L/lang_test_oracle_LG exp/tri2b \
#    $graph_dir >& $graph_dir/mkgraph.log
#
#  steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
#    exp/tri2b/decode_eval_oracle_LG_$L &
#done
#wait

# Decode with oracle G
for L in $SBS_LANGUAGES; do
  graph_dir=exp/tri3b/$L/graph_oracle_LG
  mkdir -p $graph_dir
  utils/mkgraph.sh data/$L/lang_test_oracle_LG exp/tri3b \
    $graph_dir >& $graph_dir/mkgraph.log

  #steps/decode_fmllr.sh --beam 20 --lattice-beam 16 --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/dev \
  #  exp/tri3b/decode_dev_oracle_LG_$L &
    
  steps/decode_fmllr.sh --stage 10 --beam 20 --lattice-beam 16 --nj 4 \
    --scoring-opts "--min-lmwt 1 --max-lmwt 20" \
    --cmd "$decode_cmd" $graph_dir data/$L/eval \
    exp/tri3b/decode_eval_oracle_LG_$L &
done
wait

