#!/bin/bash -u

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

# Set the language codes for SBS languages that we will be processing
#export SBS_LANGUAGES="AR DT MD HG SW UR"
export SBS_LANGUAGES="MD"

:<<'END'
#### LANGUAGE SPECIFIC SCRIPTS HERE ####
local/sbs_data_prep.sh --config-dir=$PWD/conf --corpus-dir=$SBS_CORPUS \
  --languages="$SBS_LANGUAGES"  --trans-dir=$SBS_TRANSCRIPTS --list-dir=$SBS_DATA_LISTS || exit 1;

#local/sbs_dict_prep.sh $SBS_LANGUAGES >& data/$L/prepare_dict.log || exit 1;

for L in $SBS_LANGUAGES; do
  local/sbs_dict_prep.sh $SBS_LANGUAGES >& data/$L/prepare_dict.log || exit 1;

  utils/prepare_lang.sh --position-dependent-phones false \
    data/$L/local/dict "<unk>" data/$L/local/lang_tmp data/$L/lang \
    >& data/$L/prepare_lang.log || exit 1;
done

for L in $SBS_LANGUAGES; do
    local/sbs_format_phnlm.sh $L 
	>& data/$L/format_lm.log || exit 1;
done
wait

END
:<<'END'
# Make MFCC features.
for L in $SBS_LANGUAGES; do
  mfccdir=mfcc/$L
  for x in train eval; do
    ( 
      steps/make_mfcc.sh --nj 4 --cmd "$train_cmd" data/$L/$x \
	exp/$L/make_mfcc/$x $mfccdir;
      steps/compute_cmvn_stats.sh data/$L/$x exp/$L/make_mfcc/$x $mfccdir; 
    ) &
  done
done
wait;

# Training monophone models
for L in $SBS_LANGUAGES; do
  mkdir -p exp/$L/mono;
  steps/train_mono.sh --nj 8 --cmd "$train_cmd" \
    data/$L/train data/$L/lang exp/$L/mono >& exp/$L/mono/train.log &
done
wait;

# Training/decoding monophone models
for L in $SBS_LANGUAGES; do
      graph_dir=exp/$L/mono/graph
      mkdir -p $graph_dir
	utils/mkgraph.sh --mono data/$L/lang_test exp/$L/mono \
	$graph_dir >& $graph_dir/mkgraph.log

      	steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
		exp/$L/mono/decode_eval
done


# Training/decoding triphone models
for L in $SBS_LANGUAGES; do
  (
    mkdir -p exp/$L/mono_ali
    steps/align_si.sh --nj 8 --cmd "$train_cmd" \
      data/$L/train data/$L/lang exp/$L/mono exp/$L/mono_ali \
      >& exp/$L/mono_ali/align.log 

    mkdir -p exp/$L/tri1
	steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2000 10000 \
	data/$L/train data/$L/lang exp/$L/mono_ali exp/$L/tri1 >& exp/$L/tri1/train.log || exit 1;

    ) &
done
wait;

# Training triphone models with MFCC+deltas+double-deltas
for L in $SBS_LANGUAGES; do
      graph_dir=exp/$L/tri1/graph
      mkdir -p $graph_dir
      
	utils/mkgraph.sh data/$L/lang_test exp/$L/tri1 $graph_dir \
	>& $graph_dir/mkgraph.log 

      steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
	exp/$L/tri1/decode_eval
done

for L in $SBS_LANGUAGES; do
  (
    mkdir -p exp/$L/tri1_ali
    steps/align_si.sh --nj 8 --cmd "$train_cmd" \
      data/$L/train data/$L/lang exp/$L/tri1 exp/$L/tri1_ali \
      >& exp/$L/tri1_ali/align.log 

    mkdir -p exp/$L/tri2
	steps/train_deltas.sh --cmd "$train_cmd" 2500 15000 \
	data/$L/train data/$L/lang exp/$L/tri1_ali exp/$L/tri2a || exit 1;

    ) &
done
wait;

for L in $SBS_LANGUAGES; do
      graph_dir=exp/$L/tri2a/graph
      mkdir -p $graph_dir
      
	  utils/mkgraph.sh data/$L/lang_test exp/$L/tri2a $graph_dir \
	>& $graph_dir/mkgraph.log 

      steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
	exp/$L/tri2a/decode_eval
done

# Train with LDA+MLLT transforms
for L in $SBS_LANGUAGES; do
  (
	mkdir -p exp/$L/tri2b
	steps/train_lda_mllt.sh --cmd "$train_cmd" \
	--splice-opts "--left-context=3 --right-context=3" 2500 15000 \
	data/$L/train data/$L/lang exp/$L/tri1_ali exp/$L/tri2b || exit 1;

    ) &
done
wait;

for L in $SBS_LANGUAGES; do
      graph_dir=exp/$L/tri2b/graph
      mkdir -p $graph_dir
      
	  utils/mkgraph.sh data/$L/lang_test exp/$L/tri2b $graph_dir \
	>& $graph_dir/mkgraph.log 

      steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
	exp/$L/tri2b/decode_eval
done

END
:<<'END'
# Training SAT+LDA+MLLT triphone systems
for L in $SBS_LANGUAGES; do
    #mkdir -p exp/tri2b_ali
	#steps/align_si.sh --nj 8 --cmd "$train_cmd" \
	#	data/$L/train data/$L/lang exp/$L/tri2b exp/$L/tri2b_ali \
	#	>& exp/$L/tri2b_ali/align.log 
	
	#steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
	steps/train_sat.sh --cmd "$train_cmd" 1200 8000 \
		data/train data/lang exp/tri2b exp/tri3b_2 || exit 1;

  echo ------------------------------------------
  graph_dir=exp/tri3b_2/graph
  mkdir -p $graph_dir
  utils/mkgraph.sh data/$L/lang_test_oracle_G exp/tri3b_2 $graph_dir \
    || exit 1;

  steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/eval \
    exp/tri3b_2/decode_eval_oracle_G_$L || exit 1;
  echo ------------------------------------------
done
END

# 07/15/2015
#---------------- train on train data ----------------------
# Training SAT+LDA+MLLT triphone systems
for L in $SBS_LANGUAGES; do

:<<'END'
  mkdir -p exp/$L/tri2b_ali_train_post
	
	#steps/align_si.sh --nj 8 --cmd "$train_cmd" \
	steps/align_si_post.sh --nj 8 --cmd "$train_cmd" \
		data/$L/train data/$L/lang exp/$L/tri2b exp/$L/tri2b_ali_train_post \
		>& exp/$L/tri2b_ali_train_post/align.log 
END
:<<'END'
  mkdir -p exp/tri3b_2_ali
	steps/align_fmllr.sh --nj 8 --cmd "$train_cmd" \
		data/train data/lang exp/tri3b_2 exp/tri3b_2_ali \
		 || exit 1;
		#>& exp/tri3b_ali/align.log || exit 1;
  echo ------------------------------------------
END

:<<'END'
  # align pt of target language
  mkdir -p exp/tri3b_ali_${L}_train_pt-2
	#steps/align_fmllr_post.sh --nj 8 --cmd "$train_cmd" \
	steps/align_fmllr_post.sh --stage 3 --nj 8 --cmd "$train_cmd" \
		data/$L/train data/lang exp/tri3b_2 exp/tri3b_ali_${L}_train_pt-2 \
    || exit 1;
  echo ------------------------------------------
END

:<<'END'
  # align dt of target language
  mkdir -p exp/tri3b_ali_${L}_train
	steps/align_fmllr_post_test.sh --nj 8 --cmd "$train_cmd" \
		data/$L/train data/lang exp/tri3b_2 exp/tri3b_ali_${L}_train \
		 || exit 1;
	#	>& exp/tri3b_ali_${L}_train/align.log  || exit 1;
  echo ------------------------------------------
END
	
  ##
	#steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
	#steps/train_sat_extra.sh --stage 35 --cmd "$train_cmd" 2500 15000 \
	#steps/train_sat_extra.sh --cmd "$train_cmd" 2500 15000 \
	#steps/train_sat_extra_post_test_1.sh --cmd "$train_cmd" 2500 15000 \
	#steps/train_sat_extra_post.sh --cmd "$train_cmd" 2500 15000 \
	steps/train_sat_extra_post.sh --cmd "$train_cmd" 1200 8000 \
		data/train data/$L/train data/lang exp/tri3b_2_ali \
    exp/tri3b_ali_${L}_train_pt-2 exp/tri3b_extra_${L}_train_pt-2 || exit 1;
    #exp/tri3b_ali_train_post exp/tri3b_extra_train || exit 1;
  echo ------------------------------------------

  graph_dir=exp/tri3b_extra_${L}_train_pt-2/graph_oracle_G
  mkdir -p $graph_dir
  #utils/mkgraph.sh data/$L/lang_test exp/tri3b_extra_train $graph_dir \
  utils/mkgraph.sh data/$L/lang_test_oracle_G exp/tri3b_extra_${L}_train_pt-2 $graph_dir \
    >& $graph_dir/mkgraph.log || exit 1;
  echo ------------------------------------------

  steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/eval \
    exp/tri3b_extra_${L}_train_pt-2/decode_eval || exit 1;
  echo ------------------------------------------
:<<'END'
END

:<<'END'
  graph_dir=exp/tri3b_extra_train/graph
  mkdir -p $graph_dir
  #utils/mkgraph.sh data/$L/lang_test exp/tri3b_extra_train $graph_dir \
  utils/mkgraph.sh data/$L/lang_test_oracle_G exp/tri3b_extra_train $graph_dir \
    >& $graph_dir/mkgraph.log

  steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
    exp/tri3b_extra_train/decode_eval
END
done
#---------------- train on eval data ----------------------
:<<'END'
# Training SAT+LDA+MLLT triphone systems
for L in $SBS_LANGUAGES; do
  (
    mkdir -p exp/$L/tri3b_ali_eval
	
	steps/align_fmllr.sh --nj 4 --cmd "$train_cmd" \
		data/$L/eval data/$L/lang exp/$L/tri3b exp/$L/tri3b_ali_eval \
		>& exp/$L/tri3b_ali_eval/align.log 
	
	steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
		data/$L/eval data/$L/lang exp/$L/tri3b exp/$L/tri3b_eval || exit 1;
  ) &
done
END
#------------------------------------



:<<'END'
for L in $SBS_LANGUAGES; do
      graph_dir=exp/$L/tri3b_2/graph
      mkdir -p $graph_dir
	  utils/mkgraph.sh data/$L/lang_test exp/$L/tri3b_2 $graph_dir \
	>& $graph_dir/mkgraph.log

      steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
	exp/$L/tri3b_2/decode_eval
done
END


# Getting PER numbers
# for x in exp/*/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
