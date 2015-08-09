#!/bin/bash -ue

# After training an SAT gmm-hmm system with multilingual languages 
# (among Arabic, Dutch, Mandarin, Hungarian, Swahili, Urdu) of the SBS corpus,
# proceed to adapt gmm-hmm with probabilistic transcription of target language.

echo `date` && echo $0 

[ -f cmd.sh ] && source ./cmd.sh \
  || echo "cmd.sh not found. Jobs may not execute properly."

. path.sh || { echo "Cannot source path.sh"; exit 1; }

stage=4

LANG="SW"

# add the directory of raw pt
dir_raw_pt=/export/ws15-pt-data2/data/pt-stable-7/held-out-$LANG

# prune PTs to make the compile-train-graphs stage computationally feasible;
# this value can be tuned on the dev set
prune_wt=1 

dir_fsts=exp/data_pt

feats_nj=4
train_nj=8
decode_nj=4

# # generate alignment for training data if needed
# 
# if [ $stage -le 0 ]; then
#   mkdir -p exp/tri3b_ali
#   steps/align_fmllr.sh --nj "$train_nj" --cmd "$train_cmd" \
#     data/train data/lang exp/tri3b exp/tri3b_ali
# fi

# Post-process the raw probabilistic lattice/sausages, i.e., pt, and the main
# change is to map <eps>:<eps> to <#0>:<eps>.
# Probably need to make corresponding change to this stage to process 
# different raw pt lattices. Here is an eg.

if [ $stage -le 1 ]; then
  echo "Pruning PTs"

  mkdir -p $dir_fsts
  disambig_sym=`grep "#0" data/lang/words.txt | awk '{print $2}'`

  for f in $dir_raw_pt/*lat.fst; do
    fstprint $f |  awk -v sym=$disambig_sym '{if (NF > 3 && $3 == 0) {$3 = sym}; print}' | \
      fstcompile | fstprune --weight=$prune_wt > $dir_fsts/${f##/*/}
  done
fi

# Adapting SAT+LDA+MLLT triphone systems

if [ $stage -le 2 ]; then
  echo "Aligning PTs"

  # align pt of target language
  #
  # $dir_fsts needs to be properly assigned and contains the processed pt

  if [ -z $dir_fsts ]; then echo "empty $dir_fsts" && exit 1; fi

  mkdir -p exp/tri3b_ali_${LANG}_pt
  local/align_fmllr_pt.sh --nj "$train_nj" --cmd "$train_cmd" data/$LANG/train \
    data/lang exp/tri3b exp/tri3b_ali_${LANG}_pt $dir_fsts
fi

exp_dir=exp/tri3b_map_${LANG}_pt

if [ $stage -le 3 ]; then
  echo "Adapting to PTs"

  local/train_sat_map_pt.sh --cmd "$train_cmd" \
    data/$LANG/train data/lang exp/tri3b_ali_${LANG}_pt $exp_dir
fi

if [ $stage -le 4 ]; then
  echo "Decoding"

  graph_dir=$exp_dir/graph_text_G
  mkdir -p $graph_dir
  utils/mkgraph.sh data/$LANG/lang_test_text_G $exp_dir $graph_dir

  steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" $graph_dir \
    data/$LANG/dev $exp_dir/decode_dev
fi

echo `date`

#---------------------------------------------------------------------------
# Getting PER numbers
# for x in exp/*/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done


