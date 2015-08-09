#!/bin/bash -e

#
# Commands below are supposed to be run after run.sh.
# 

echo "This shell script may run as-is on your system, but it is recommended
that you run the commands one by one by copying and pasting into the shell."
# exit 1

set -e

. ./cmd.sh
. ./path.sh

SBS_LANGUAGES="AR CA DT HG MD SW UR"
TEXT_PHONE_LM=/export/ws15-pt-data/data/text-phnlm

. utils/parse_options.sh

for L in $SBS_LANGUAGES; do
  echo "Prep oracle G for $L"
  local/sbs_format_text_G.sh --text-phone-lm $TEXT_PHONE_LM $L
done

# Decode with oracle G: mono
for L in $SBS_LANGUAGES; do
  graph_dir=exp/mono/$L/graph_text_G
  mkdir -p $graph_dir
  utils/mkgraph.sh --mono data/$L/lang_test_text_G exp/mono \
    $graph_dir >& $graph_dir/mkgraph.log

  steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/dev \
    exp/mono/decode_dev_text_G_$L &
done
wait

# Decode with oracle G: tri1 
for L in $SBS_LANGUAGES; do
  graph_dir=exp/tri1/$L/graph_text_G
  mkdir -p $graph_dir
  utils/mkgraph.sh data/$L/lang_test_text_G exp/tri1 \
    $graph_dir >& $graph_dir/mkgraph.log

  steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/dev \
    exp/tri1/decode_dev_text_G_$L &
done
wait

# Decode with oracle G: tri2b
for L in $SBS_LANGUAGES; do
  graph_dir=exp/tri2b/$L/graph_text_G
  mkdir -p $graph_dir
  utils/mkgraph.sh data/$L/lang_test_text_G exp/tri2b \
    $graph_dir >& $graph_dir/mkgraph.log

  steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/dev \
    exp/tri2b/decode_dev_text_G_$L &
done
wait

# Decode with oracle G: tri3b
for L in $SBS_LANGUAGES; do
  graph_dir=exp/tri3b/$L/graph_text_G
  mkdir -p $graph_dir
  utils/mkgraph.sh data/$L/lang_test_text_G exp/tri3b \
    $graph_dir >& $graph_dir/mkgraph.log

  steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/dev \
    exp/tri3b/decode_dev_text_G_$L &
done
wait
