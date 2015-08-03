#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.


# This does Speaker Adapted Training (SAT), i.e. train on
# fMLLR-adapted features.  It can be done on top of either LDA+MLLT, or
# delta and delta-delta features.  If there are no transforms supplied
# in the alignment directory, it will estimate transforms itself before
# building the tree (and in any case, it estimates transforms a number
# of times during training).


# Begin configuration section.
stage=-5
exit_stage=-100 # you can use this to require it to exit at the
                # beginning of a specific stage.  Not all values are
                # supported.
fmllr_update_type=full
cmd=run.pl
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10
#retry_beam=40
retry_beam=500
careful=false
boost_silence=1.0 # Factor by which to boost silence likelihoods in alignment
context_opts=  # e.g. set this to "--context-width 5 --central-position 2" for quinphone.
realign_iters="10 20 30";
fmllr_iters="2 4 6 12";
silence_weight=0.0 # Weight on silence in fMLLR estimation.
num_iters=35   # Number of iterations of training
max_iter_inc=25 # Last iter to increase #Gauss on.
power=0.2 # Exponent for number of gaussians according to occurrence counts
cluster_thresh=-1  # for build-tree control final bottom-up clustering of leaves
phone_map=
train_tree=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

#if [ $# != 6 ]; then
if [ $# != 8 ]; then
  echo "Usage: steps/train_sat.sh <#leaves> <#gauss> <data> <lang> <ali-dir> <exp-dir>"
  echo " e.g.: steps/train_sat.sh 2500 15000 data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri3b"
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  exit 1;
fi

:<<'END'
dir_fsts="/export/ws15-pt-data/data/phonelattices/monophones/engg2p/mandarin/"
d=/export/ws15-pt-data/cliu/kaldi-trunk/egs/SBS-try1
$cmd JOB=1:1 $d/"head22" \
  ls $dir_fsts \| awk -v dir_fsts="$dir_fsts" '{key=value=$1; gsub(/.saus.fst/, "", key); print key"\t"dir_fsts""value}'
echo `date` && exit 0;
END

numleaves=$1
totgauss=$2
data_1=$3
data_2=$4
lang=$5
alidir_1=$6
alidir_2=$7
dir_exp=$8
dir_1=$dir_exp"_1"
dir_2=$dir_exp"_2"

for f in $data_1/feats.scp $lang/phones.txt $alidir_1/final.mdl $alidir_1/ali.1.gz \
  $data_2/feats.scp $alidir_2/final.mdl $alidir_2/post.1.gz; do
  [ ! -f $f ] && echo "train_sat.sh: no such file $f" && exit 1;
done

numgauss=$numleaves
incgauss=$[($totgauss-$numgauss)/$max_iter_inc]  # per-iter #gauss increment
oov=`cat $lang/oov.int`
nj=`cat $alidir_1/num_jobs` || exit 1;
[[ $nj != `cat $alidir_2/num_jobs` ]] && echo "nj mismatch" && exit 1;
silphonelist=`cat $lang/phones/silence.csl`
ciphonelist=`cat $lang/phones/context_indep.csl` || exit 1;
sdata_1=$data_1/split$nj;
sdata_2=$data_2/split$nj;
splice_opts=`cat $alidir_1/splice_opts 2>/dev/null` # frame-splicing options.
cmvn_opts=`cat $alidir_1/cmvn_opts 2>/dev/null`
delta_opts=`cat $alidir_1/delta_opts 2>/dev/null`
phone_map_opt=
[ ! -z "$phone_map" ] && phone_map_opt="--phone-map='$phone_map'"

for dir in $dir_exp $dir_1 $dir_2; do
  mkdir -p $dir/log
  cp $alidir_1/splice_opts $dir 2>/dev/null # frame-splicing options.
  cp $alidir_1/cmvn_opts $dir 2>/dev/null # cmn/cmvn option.
  cp $alidir_1/delta_opts $dir 2>/dev/null # delta option.

  echo $nj >$dir/num_jobs
done

##
cp $alidir_1/final.mat $dir_exp
cp $alidir_1/full.mat $dir_exp 2>/dev/null
#
for i in 1; do
  eval sdata_i=\$sdata_$i
  eval data_i=\$data_$i
  eval alidir_i=\$alidir_$i
  eval dir_i=\$dir_$i
  #echo $sdata_i $data_i $alidir_i $dir_i; echo `date` && exit 0;

  [[ -d $sdata_i && $data_i/feats.scp -ot $sdata_i ]] || split_data.sh $data_i $nj || exit 1;

  # Set up features.
  if [ -f $alidir_i/final.mat ]; then feat_type=lda; else feat_type=delta; fi
  echo "$0: feature type is $feat_type"

  ## Set up speaker-independent features.
  case $feat_type in
    delta) declare sifeats_$i="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata_i/JOB/utt2spk scp:$sdata_i/JOB/cmvn.scp scp:$sdata_i/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |";;
    lda) declare sifeats_$i="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata_i/JOB/utt2spk scp:$sdata_i/JOB/cmvn.scp scp:$sdata_i/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir_i/final.mat ark:- ark:- |"
      cp $alidir_i/final.mat $dir_i
      cp $alidir_i/full.mat $dir_i 2>/dev/null
      ;;
    *) echo "$0: invalid feature type $feat_type" && exit 1;
  esac
  eval sifeats_i=\$sifeats_$i
  #echo "line 98 "; echo $sifeats_i; echo ""


  ## Get initial fMLLR transforms (possibly from alignment dir)
  if [ -f $alidir_i/trans.1 ]; then
    echo "$0: Using transforms from $alidir_i"
    declare feats_$i="$sifeats_i transform-feats --utt2spk=ark:$sdata_i/JOB/utt2spk ark,s,cs:$alidir_i/trans.JOB ark:- ark:- |"
    declare cur_trans_dir_$i=$alidir_i
  else 
    if [ $stage -le -5 ]; then
      echo "$0: obtaining initial fMLLR transforms since not present in $alidir_i"
      # The next line is necessary because of $silphonelist otherwise being incorrect; would require
      # old $lang dir which would require another option.  Not needed anyway.
      [ ! -z "$phone_map" ] && \
         echo "$0: error: you must provide transforms if you use the --phone-map option." && exit 1;
      $cmd JOB=1:$nj $dir_i/log/fmllr.0.JOB.log \
        ali-to-post "ark:gunzip -c $alidir_i/ali.JOB.gz|" ark:- \| \
        weight-silence-post $silence_weight $silphonelist $alidir_i/final.mdl ark:- ark:- \| \
        gmm-est-fmllr --fmllr-update-type=$fmllr_update_type \
        --spk2utt=ark:$sdata_i/JOB/spk2utt $alidir_i/final.mdl "$sifeats_i" \
        ark:- ark:$dir_i/trans.JOB || exit 1;
    fi
    declare feats_$i="$sifeats_i transform-feats --utt2spk=ark:$sdata_i/JOB/utt2spk ark,s,cs:$dir_i/trans.JOB ark:- ark:- |"
    declare cur_trans_dir_$i=$dir_i
  fi
  #eval echo \$feats_$i \$cur_trans_dir_$i; echo `date` && exit 0;
done
#
for i in 2; do
  eval sdata_i=\$sdata_$i
  eval data_i=\$data_$i
  eval alidir_i=\$alidir_$i
  eval dir_i=\$dir_$i
  #echo $sdata_i $data_i $alidir_i $dir_i; echo `date` && exit 0;

  [[ -d $sdata_i && $data_i/feats.scp -ot $sdata_i ]] || split_data.sh $data_i $nj || exit 1;

  # Set up features.
  if [ -f $alidir_i/final.mat ]; then feat_type=lda; else feat_type=delta; fi
  echo "$0: feature type is $feat_type"

  ## Set up speaker-independent features.
  case $feat_type in
    delta) declare sifeats_$i="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata_i/JOB/utt2spk scp:$sdata_i/JOB/cmvn.scp scp:$sdata_i/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |";;
    lda) declare sifeats_$i="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata_i/JOB/utt2spk scp:$sdata_i/JOB/cmvn.scp scp:$sdata_i/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir_i/final.mat ark:- ark:- |"
      cp $alidir_i/final.mat $dir_i
      cp $alidir_i/full.mat $dir_i 2>/dev/null
      ;;
    *) echo "$0: invalid feature type $feat_type" && exit 1;
  esac
  eval sifeats_i=\$sifeats_$i
  #echo "line 98 "; echo $sifeats_i; echo ""


  ## Get initial fMLLR transforms (possibly from alignment dir)
  if [ -f $alidir_i/trans.1 ]; then
    echo "$0: Using transforms from $alidir_i"
    declare feats_$i="$sifeats_i transform-feats --utt2spk=ark:$sdata_i/JOB/utt2spk ark,s,cs:$alidir_i/trans.JOB ark:- ark:- |"
    declare cur_trans_dir_$i=$alidir_i
  else 
    if [ $stage -le -5 ]; then
      echo "$0: obtaining initial fMLLR transforms since not present in $alidir_i"
      # The next line is necessary because of $silphonelist otherwise being incorrect; would require
      # old $lang dir which would require another option.  Not needed anyway.
      [ ! -z "$phone_map" ] && \
         echo "$0: error: you must provide transforms if you use the --phone-map option." && exit 1;
      #ali-to-post "ark:gunzip -c $alidir_i/ali.JOB.gz|" ark:- \| \
      $cmd JOB=1:$nj $dir_i/log/fmllr.0.JOB.log \
        weight-silence-post $silence_weight $silphonelist $alidir_i/final.mdl \
        "ark:gunzip -c $alidir_i/post.JOB.gz|" ark:- \| \
        gmm-est-fmllr --fmllr-update-type=$fmllr_update_type \
        --spk2utt=ark:$sdata_i/JOB/spk2utt $alidir_i/final.mdl "$sifeats_i" \
        ark:- ark:$dir_i/trans.JOB || exit 1;
    fi
    declare feats_$i="$sifeats_i transform-feats --utt2spk=ark:$sdata_i/JOB/utt2spk ark,s,cs:$dir_i/trans.JOB ark:- ark:- |"
    declare cur_trans_dir_$i=$dir_i
  fi
  #eval echo \$feats_$i \$cur_trans_dir_$i; echo `date` && exit 0;
done

:<<'END'
#if [ $stage -le -4 ] && $train_tree && [ $i == "1" ]; then
if [ $stage -le -4 ] && $train_tree; then
  # Get tree stats.
  echo "$0: Accumulating tree stats"
  #eval feats_i=\$feats_$i; #echo $feats_i && exit 0;
  $cmd JOB=1:$nj $dir_1/log/acc_tree.JOB.log \
    acc-tree-stats $context_opts $phone_map_opt --ci-phones=$ciphonelist $alidir_1/final.mdl "$feats_1" \
    "ark:gunzip -c $alidir_1/ali.JOB.gz|" $dir_1/JOB.treeacc || exit 1;
  [ "`ls $dir_1/*.treeacc | wc -w`" -ne "$nj" ] && echo "$0: Wrong #tree-accs" && exit 1;
  $cmd $dir_1/log/sum_tree_acc.log \
    sum-tree-stats $dir_1/treeacc $dir_1/*.treeacc || exit 1;
  rm $dir_1/*.treeacc
fi
  #echo "line 122 "; echo $feats_i; echo ""
#done
##
#echo `date` && exit 0;

if [ $stage -le -4 ] && $train_tree; then
  $cmd $dir_exp/log/sum_tree_acc.log \
    sum-tree-stats $dir_exp/treeacc $dir_1/treeacc || exit 1; #echo `date` && exit 0;
    #sum-tree-stats $dir_exp/treeacc $dir_1/treeacc $dir_2/treeacc || exit 1; #echo `date` && exit 0;
fi

if [ $stage -le -3 ] && $train_tree; then
  echo "$0: Getting questions for tree clustering."
  # preparing questions, roots file...
  cluster-phones $context_opts $dir_exp/treeacc $lang/phones/sets.int $dir_exp/questions.int 2> $dir_exp/log/questions.log || exit 1;
  cat $lang/phones/extra_questions.int >> $dir_exp/questions.int
  compile-questions $context_opts $lang/topo $dir_exp/questions.int $dir_exp/questions.qst 2>$dir_exp/log/compile_questions.log || exit 1;

  echo "$0: Building the tree"
  $cmd $dir_exp/log/build_tree.log \
    build-tree $context_opts --verbose=1 --max-leaves=$numleaves \
    --cluster-thresh=$cluster_thresh $dir_exp/treeacc $lang/phones/roots.int \
    $dir_exp/questions.qst $lang/topo $dir_exp/tree || exit 1;
fi
END
if [ $stage -le -4 ]; then
  cp $alidir_1/tree $dir_exp/tree
  #cp exp/MD/tri3b_2/treeacc $dir_exp/treeacc #
  cp ${alidir_1%_ali}/treeacc $dir_exp/treeacc #
  echo "use the same treeacc by \"cp ${alidir_1%_ali}/treeacc $dir_exp/treeacc\""
  #cp $alidir_1/questions.int $dir_exp/questions.int
  #cp $alidir_1/questions.qst $dir_exp/questions.qst
fi

#echo `date` && exit 0;

if [ $stage -le -2 ]; then
  echo "$0: Initializing the model"
  if $train_tree; then
    gmm-init-model  --write-occs=$dir_exp/1.occs  \
      $dir_exp/tree $dir_exp/treeacc $lang/topo $dir_exp/1.mdl 2> $dir_exp/log/init_model.log || exit 1;
    grep 'no stats' $dir_exp/log/init_model.log && echo "This is a bad warning.";
    rm $dir_exp/treeacc
  else
    echo "Should have used treeacc" && exit 1;
    cp $alidir_1/tree $dir_exp/ || exit 1;
    $cmd JOB=1 $dir_exp/log/init_model.log \
      gmm-init-model-flat $dir_exp/tree $lang/topo $dir_exp/1.mdl \
        "$feats subset-feats ark:- ark:-|" || exit 1;
  fi
fi

#:<<'END'
if [ $stage -le -1 ]; then
  #for i in 1 2; do
  for i in 1; do
    eval dir_i=\$dir_$i
    eval alidir_i=\$alidir_$i
    # Convert the alignments.
    echo "$0: Converting alignments from $alidir_i to use current tree"
    $cmd JOB=1:$nj $dir_i/log/convert.JOB.log \
      convert-ali $phone_map_opt $alidir_i/final.mdl $dir_exp/1.mdl $dir_exp/tree \
       "ark:gunzip -c $alidir_i/ali.JOB.gz|" "ark:|gzip -c >$dir_i/ali.JOB.gz" || exit 1;
  done
  for i in 2; do
    $cmd JOB=1:$nj $dir_2/log/convert.JOB.log \
      cp $alidir_2/post.JOB.gz $dir_2/ || exit 1;
  done
fi
#END

[ "$exit_stage" -eq 0 ] && echo "$0: Exiting early: --exit-stage $exit_stage" && exit 0;

if [ $stage -le 0 ] && [ "$realign_iters" != "" ]; then
  for i in 1; do
    eval dir_i=\$dir_$i
    eval sdata_i=\$sdata_$i
    cp $dir_exp/tree $dir_i/ || exit 1; #echo $dir_i  $sdata_i
    cp $dir_exp/1.mdl $dir_i/ || exit 1;
    echo "$0: Compiling graphs of transcripts"
    $cmd JOB=1:$nj $dir_i/log/compile_graphs.JOB.log \
      compile-train-graphs $dir_i/tree $dir_i/1.mdl  $lang/L.fst  \
       "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $sdata_i/JOB/text |" \
        "ark:|gzip -c >$dir_i/fsts.JOB.gz" || exit 1;
  done

  ##
  cp $dir_exp/tree $dir_2/ || exit 1;
  cp $dir_exp/1.mdl $dir_2/ || exit 1;
  cp $alidir_2/fsts.*.gz $dir_2/ || exit 1;
  echo "$0: Compiling graphs of transcripts"
:<<'END'
  #dir_fsts="/export/ws15-pt-data/data/phonelattices/monophones/engg2p/mandarin/"
  #ls $dir_fsts/*saus.fst | awk -v dir_fsts=$dir_fsts '{key=value=$1; gsub(/.saus.fst/, "", key); print key"\t"dir_fsts""value}' > $data_2/pt.txt
  #dir_fsts="/export/ws15-pt-data/cliu/data/phonelattices/monophones/engg2p/mandarin/"
  dir_fsts="/export/ws15-pt-data/cliu/data/phonelattices/monophones/engg2p/mandarin-no_epsilon_1/"
  $cmd JOB=1:$nj $dir_2/log/text_pt.JOB.log \
    cut -f1 "$sdata_2/JOB/text" \| awk -v dir_fsts=$dir_fsts \
      '{key=value=$1; value=value".saus.fst"; print key"\t"dir_fsts""value}' \
      \> "$sdata_2/JOB/text.pt"

  $cmd JOB=1:$nj $dir_2/log/compile_graphs.JOB.log \
    compile-train-graphs-fsts-pt --read-disambig-syms=$lang/phones/disambig.int \
    --batch-size=1 --transition-scale=1.0 --self-loop-scale=0.1 $dir_2/tree $dir_2/1.mdl  $lang/L_disambig.fst  \
     "ark:cat $sdata_2/JOB/text.pt|" "ark:|gzip -c >$dir_2/fsts.JOB.gz" || exit 1;
    #--batch-size=1 $dir_2/tree $dir_2/1.mdl  $lang/L.fst  \
END
fi

#echo `date` line 232 && exit 0;

x=1; #x=35
while [ $x -lt $num_iters ]; do
   echo Pass $x
  if echo $realign_iters | grep -w $x >/dev/null && [ $stage -le $x ]; then
    echo Aligning data
    mdl="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $dir_exp/$x.mdl - |"
    #for i in 1 2; do
    for i in 1; do
      eval dir_i=\$dir_$i
      eval feats_i=\$feats_$i
      $cmd JOB=1:$nj $dir_i/log/align.$x.JOB.log \
        gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam --careful=$careful "$mdl" \
        "ark:gunzip -c $dir_i/fsts.JOB.gz|" "$feats_i" \
        "ark:|gzip -c >$dir_i/ali.JOB.gz" || exit 1;
    done

    # align pt data
    for i in 2; do
      eval dir_i=\$dir_$i
      eval feats_i=\$feats_$i
      #maxactive=7000; beam_=20.0; lattice_beam=7.0; acwt=0.083333;
      maxactive=7000; beam_=20.0; lattice_beam=3.0; acwt=0.083333;
      #maxactive=80000; beam_=700.0; lattice_beam=3.0; acwt=0.083333;
      #maxactive=50000; beam_=400.0; lattice_beam=3.0; acwt=0.083333;
      $cmd JOB=1:$nj $dir_i/log/align.$x.JOB.log \
        gmm-latgen-faster --max-active=$maxactive --beam=$beam_ --lattice-beam=$lattice_beam --acoustic-scale=$acwt \
          --allow-partial=true --word-symbol-table=$lang/words.txt \
          $dir_exp/$x.mdl "ark:gunzip -c $dir_i/fsts.JOB.gz|" "$feats_i" ark:- \| \
          lattice-to-post --acoustic-scale=$acwt ark:- "ark:|gzip -c >$dir_i/post.JOB.gz"
          # weight-silence-post 0.0 $silphonelist $srcdir/final.alimdl ark:- ark:-
          #gmm-acc-stats 1.mdl scp:train.scp ark:- 1.acc
    done
  fi

  if echo $fmllr_iters | grep -w $x >/dev/null; then
    if [ $stage -le $x ]; then
      echo Estimating fMLLR transforms
      # We estimate a transform that's additional to the previous transform;
      # we'll compose them.
      #for i in 1 2; do
      for i in 1; do
        eval dir_i=\$dir_$i
        eval sdata_i=\$sdata_$i
        eval feats_i=\$feats_$i; #echo "line 217 "; echo $feats_i; echo ""
        eval cur_trans_dir_i=\$cur_trans_dir_$i; #echo $cur_trans_dir_i
        $cmd JOB=1:$nj $dir_i/log/fmllr.$x.JOB.log \
          ali-to-post "ark:gunzip -c $dir_i/ali.JOB.gz|" ark:-  \| \
          weight-silence-post $silence_weight $silphonelist $dir_exp/$x.mdl ark:- ark:- \| \
          gmm-est-fmllr --fmllr-update-type=$fmllr_update_type \
          --spk2utt=ark:$sdata_i/JOB/spk2utt $dir_exp/$x.mdl \
          "$feats_i" ark:- ark:$dir_i/tmp_trans.JOB || exit 1;
        for n in `seq $nj`; do
          ! ( compose-transforms --b-is-affine=true \
            ark:$dir_i/tmp_trans.$n ark:$cur_trans_dir_i/trans.$n ark:$dir_i/composed_trans.$n \
            && mv $dir_i/composed_trans.$n $dir_i/trans.$n && \
            rm $dir_i/tmp_trans.$n ) 2>$dir_i/log/compose_transforms.$x.log \
            && echo "$0: Error composing transforms" && exit 1;
        done
        eval sifeats_i=\$sifeats_$i
        declare feats_$i="$sifeats_i transform-feats --utt2spk=ark:$sdata_i/JOB/utt2spk ark:$dir_i/trans.JOB ark:- ark:- |"
        declare cur_trans_dir_$i=$dir_i
      done

      ##
      for i in 2; do
        eval dir_i=\$dir_$i
        eval sdata_i=\$sdata_$i
        eval feats_i=\$feats_$i; #echo "line 217 "; echo $feats_i; echo ""
        eval cur_trans_dir_i=\$cur_trans_dir_$i; #echo $cur_trans_dir_i
        $cmd JOB=1:$nj $dir_i/log/fmllr.$x.JOB.log \
          weight-silence-post $silence_weight $silphonelist $dir_exp/$x.mdl \
          "ark:gunzip -c $dir_i/post.JOB.gz|" ark:- \| \
          gmm-est-fmllr --fmllr-update-type=$fmllr_update_type \
          --spk2utt=ark:$sdata_i/JOB/spk2utt $dir_exp/$x.mdl \
          "$feats_i" ark:- ark:$dir_i/tmp_trans.JOB || exit 1;
        for n in `seq $nj`; do
          ! ( compose-transforms --b-is-affine=true \
            ark:$dir_i/tmp_trans.$n ark:$cur_trans_dir_i/trans.$n ark:$dir_i/composed_trans.$n \
            && mv $dir_i/composed_trans.$n $dir_i/trans.$n && \
            rm $dir_i/tmp_trans.$n ) 2>$dir_i/log/compose_transforms.$x.log \
            && echo "$0: Error composing transforms" && exit 1;
        done
        eval sifeats_i=\$sifeats_$i
        declare feats_$i="$sifeats_i transform-feats --utt2spk=ark:$sdata_i/JOB/utt2spk ark:$dir_i/trans.JOB ark:- ark:- |"
        declare cur_trans_dir_$i=$dir_i
      done
    fi
  fi
  
  if [ $stage -le $x ]; then
    #for i in 1 2; do
    for i in 1; do
      eval dir_i=\$dir_$i
      eval feats_i=\$feats_$i
      $cmd JOB=1:$nj $dir_i/log/acc.$x.JOB.log \
        gmm-acc-stats-ali $dir_exp/$x.mdl "$feats_i" \
        "ark,s,cs:gunzip -c $dir_i/ali.JOB.gz|" $dir_i/$x.JOB.acc || exit 1;
      [ `ls $dir_i/$x.*.acc | wc -w` -ne "$nj" ] && echo "$0: Wrong #accs" && exit 1;
    done
    # align pt data
    for i in 2; do
      eval dir_i=\$dir_$i
      eval feats_i=\$feats_$i
      $cmd JOB=1:$nj $dir_i/log/acc.$x.JOB.log \
        gmm-acc-stats $dir_exp/$x.mdl "$feats_i" \
        "ark,s,cs:gunzip -c $dir_i/post.JOB.gz|" $dir_i/$x.JOB.acc || exit 1;
      [ `ls $dir_i/$x.*.acc | wc -w` -ne "$nj" ] && echo "$0: Wrong #accs" && exit 1;
    done

    #
    $cmd $dir_exp/log/update.$x.log \
      gmm-est --power=$power --write-occs=$dir_exp/$[$x+1].occs --mix-up=$numgauss $dir_exp/$x.mdl \
      "gmm-sum-accs - $dir_1/$x.*.acc $dir_2/$x.*.acc | gmm-scale-accs 0.5 - - |" $dir_exp/$[$x+1].mdl || exit 1;
      #"gmm-sum-accs - $dir_1/$x.*.acc $dir_2/$x.*.acc |" $dir_exp/$[$x+1].mdl || exit 1;
    #echo `date` line 292 && exit 0;
    rm $dir_exp/$x.mdl $dir_1/$x.*.acc $dir_2/$x.*.acc
    rm $dir_exp/$x.occs 
  fi
  [ $x -le $max_iter_inc ] && numgauss=$[$numgauss+$incgauss];
  x=$[$x+1];
  #echo `date` line 297 && exit 0;
done


if [ $stage -le $x ]; then
  # Accumulate stats for "alignment model"-- this model is
  # computed with the speaker-independent features, but matches Gaussian-for-Gaussian
  # with the final speaker-adapted model.
  #for i in 1 2; do
  for i in 1; do
    eval dir_i=\$dir_$i
    eval sifeats_i=\$sifeats_$i
    eval feats_i=\$feats_$i
    $cmd JOB=1:$nj $dir_i/log/acc_alimdl.JOB.log \
      ali-to-post "ark:gunzip -c $dir_i/ali.JOB.gz|" ark:-  \| \
      gmm-acc-stats-twofeats $dir_exp/$x.mdl "$feats_i" "$sifeats_i" \
      ark,s,cs:- $dir_i/$x.JOB.acc || exit 1;
    [ `ls $dir_i/$x.*.acc | wc -w` -ne "$nj" ] && echo "$0: Wrong #accs" && exit 1;
  done

  ##
  for i in 2; do
    eval dir_i=\$dir_$i
    eval sifeats_i=\$sifeats_$i
    eval feats_i=\$feats_$i
    $cmd JOB=1:$nj $dir_i/log/acc_alimdl.JOB.log \
      gmm-acc-stats-twofeats $dir_exp/$x.mdl "$feats_i" "$sifeats_i" \
      "ark,s,cs:gunzip -c $dir_i/post.JOB.gz|" $dir_i/$x.JOB.acc || exit 1;
    [ `ls $dir_i/$x.*.acc | wc -w` -ne "$nj" ] && echo "$0: Wrong #accs" && exit 1;
  done

  # Update model.
  $cmd $dir_exp/log/est_alimdl.log \
    gmm-est --power=$power --remove-low-count-gaussians=false $dir_exp/$x.mdl \
    "gmm-sum-accs - $dir_1/$x.*.acc $dir_2/$x.*.acc| gmm-scale-accs 0.5 - - |" $dir_exp/$x.alimdl  || exit 1;
    #"gmm-sum-accs - $dir_1/$x.*.acc $dir_2/$x.*.acc|" $dir_exp/$x.alimdl  || exit 1;
  rm $dir_1/$x.*.acc $dir_2/$x.*.acc
fi

rm $dir_exp/final.{mdl,alimdl,occs} 2>/dev/null
ln -s $x.mdl $dir_exp/final.mdl
ln -s $x.occs $dir_exp/final.occs
ln -s $x.alimdl $dir_exp/final.alimdl



utils/summarize_warnings.pl $dir/log
(
  echo "$0: Likelihood evolution:"
  for x in `seq $[$num_iters-1]`; do
    for i in 1 2; do
      tail -n 30 $dir_i/log/acc.$x.*.log | awk '/Overall avg like/{l += $(NF-3)*$(NF-1); t += $(NF-1); }
          /Overall average logdet/{d += $(NF-3)*$(NF-1); t2 += $(NF-1);} 
          END{ d /= t2; l /= t; printf("%s ", d+l); } '
    done
  done
  echo
) | tee $dir_exp/log/summary.log

echo Done
