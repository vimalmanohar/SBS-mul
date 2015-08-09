#!/bin/bash -e

# This script shows the steps needed to build a recognizer for certain matched languages (Arabic, Dutch, Mandarin, Hungarian, Swahili, Urdu) of the SBS corpus. 
# (Adapted from the egs/gp script run.sh)

echo "This shell script may run as-is on your system, but it is recommended 
that you run the commands one by one by copying and pasting into the shell."
#exit 1;

[ -f cmd.sh ] && source ./cmd.sh \
  || echo "cmd.sh not found. Jobs may not execute properly."

. path.sh || { echo "Cannot source path.sh"; exit 1; }

# Set the location of the SBS speech 
SBS_CORPUS=/export/ws15-pt-data/data/audio
SBS_TRANSCRIPTS=/export/ws15-pt-data/data/transcripts/matched
SBS_DATA_LISTS=/export/ws15-pt-data/data/lists
NUMLEAVES=1200
NUMGAUSSIANS=8000

# Set the language codes for SBS languages that we will be processing
export SBS_LANGUAGES="AR CA DT HG MD SW UR"

#### LANGUAGE SPECIFIC SCRIPTS HERE ####
local/sbs_data_prep.sh --config-dir=$PWD/conf --corpus-dir=$SBS_CORPUS \
  --languages="$SBS_LANGUAGES"  --trans-dir=$SBS_TRANSCRIPTS --list-dir=$SBS_DATA_LISTS

echo "dict prep"
local/sbs_dict_prep.sh $SBS_LANGUAGES

for L in $SBS_LANGUAGES; do
  echo "lang prep: $L"
  utils/prepare_lang.sh --position-dependent-phones false \
    data/$L/local/dict "<unk>" data/$L/local/lang_tmp data/$L/lang
done

for L in $SBS_LANGUAGES; do
  echo "LM prep: $L"
  local/sbs_format_phnlm.sh $L
done

echo "MFCC prep"
# Make MFCC features.
for L in $SBS_LANGUAGES; do
  mfccdir=mfcc/$L
  for x in train dev eval; do
    (
      steps/make_mfcc.sh --nj 4 --cmd "$train_cmd" data/$L/$x \
        exp/$L/make_mfcc/$x $mfccdir
      steps/compute_cmvn_stats.sh data/$L/$x exp/$L/make_mfcc/$x $mfccdir
    ) &
  done
done
wait;

for L in $SBS_LANGUAGES; do
  mkdir -p exp/$L/mono;
  steps/train_mono.sh --nj 8 --cmd "$train_cmd" \
    data/$L/train data/$L/lang exp/$L/mono
  
  graph_dir=exp/$L/mono/graph
  mkdir -p $graph_dir
  utils/mkgraph.sh --mono data/$L/lang_test exp/$L/mono $graph_dir

  graph_dir=exp/$L/mono/graph
  steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/dev \
    exp/$L/mono/decode_dev &

  # Training/decoding triphone models
  mkdir -p exp/mono_ali
  steps/align_si.sh --nj 8 --cmd "$train_cmd" \
    data/$L/train data/$L/lang exp/$L/mono exp/$L/mono_ali
  
  # Training triphone models with MFCC+deltas+double-deltas
  mkdir -p exp/$L/tri1
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" $NUMLEAVES $NUMGAUSSIANS \
    data/$L/train data/$L/lang exp/$L/mono_ali exp/$L/tri1
  
  graph_dir=exp/$L/tri1/graph
  mkdir -p $graph_dir
  
  utils/mkgraph.sh data/$L/lang_test exp/$L/tri1 $graph_dir

  graph_dir=exp/$L/tri1/graph
  steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/dev \
    exp/$L/tri1/decode_dev &

  mkdir -p exp/$L/tri1_ali
  steps/align_si.sh --nj 8 --cmd "$train_cmd" \
    data/$L/train data/$L/lang exp/$L/tri1 exp/$L/tri1_ali

  mkdir -p exp/$L/tri2b
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" $NUMLEAVES $NUMGAUSSIANS \
    data/$L/train data/$L/lang exp/$L/tri1_ali exp/$L/tri2b
  
  # Train with LDA+MLLT transforms
  graph_dir=exp/$L/tri2b/graph
  mkdir -p $graph_dir
        
  utils/mkgraph.sh data/$L/lang_test exp/$L/tri2b $graph_dir

  graph_dir=exp/$L/tri2b/graph
  steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/dev \
    exp/$L/tri2b/decode_dev &

  mkdir -p exp/$L/tri2b_ali
  
  steps/align_si.sh --nj 8 --cmd "$train_cmd" --use-graphs true \
    data/$L/train data/$L/lang exp/$L/tri2b exp/$L/tri2b_ali
  
  steps/train_sat.sh --cmd "$train_cmd" $NUMLEAVES $NUMGAUSSIANS \
    data/$L/train data/$L/lang exp/$L/tri2b exp/$L/tri3b
  
  graph_dir=exp/$L/tri3b/graph
  mkdir -p $graph_dir
  utils/mkgraph.sh data/$L/lang_test exp/$L/tri3b $graph_dir

  graph_dir=exp/$L/tri3b/graph
  steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/dev \
    exp/$L/tri3b/decode_dev &
done
wait

# Getting PER numbers
# for x in exp/*/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
