#!/bin/bash

##renew_nnet_softmax.sh
##Paul Hager

#Setting up $PATH
. ./path.sh

usage="Usage: $0 <hmm model> <input_network> <output_network> > <log_file>"
##e.g. renew_nnet_softmax.sh exp/tri3b/final.mdl nnet.old nnet.new

echo "$0 $@"  # Print the command line for logging

mdl=$1    # hmm-gmm mdl file
oldnn=$2  # old dnn
newnn=$3  # new dnn

[[ $# -eq 3 ]] || { echo $usage; exit 1; }
[[ -e $mdl ]] || { echo "$mdl does not exist"; exit 1; }
[[ -e $oldnn ]] || { echo "$oldnn does not exist"; exit 1; }

nndir=$(mktemp -d)
trap "rm -rf $nndir;" EXIT
echo
echo "Searching for number of PFIDs"
#Discover softmax dimensions from number of triphone tree leaves
softmax_dim=$(hmm-info $mdl | grep pdfs | awk '{ print $NF }')
echo "Number of PFIDs:" ${softmax_dim}

echo "Stripping pre-trained network of softmax layer"
#Remove last layer from pre-trained network
#Remove both the old softmax and the affine transformation to that layer
oldnnname=$(basename $oldnn)
nnet_stripped=$nndir/${oldnnname}_stripped.init
nnet-copy --binary=false --remove-last-components=2 $oldnn ${nnet_stripped}
echo "Done"
echo

echo "Protyping new softmax layer"
#Create softmax layer
##Note the number of hidden neurons must be greater than zero, 1 was arbitrarily chosen
tempnn="softmax"
oldnn_outdim=$(nnet-info ${nnet_stripped}|tail -n 1|awk '{print $NF}'|sed 's/\([0-9]\+\).*/\1/g')
echo "Output dim of last hidden layer = ${oldnn_outdim}"
[[ ! -z ${oldnn_outdim} ]] && [[ ${oldnn_outdim} -gt 0 ]] || { echo "output dim of the old nn is not valid = ${oldnn_outdim}"; exit 1; }
python utils/nnet/make_nnet_proto.py --no-proto-head 39 ${softmax_dim} 1 ${oldnn_outdim}|tail -n 3 > $nndir/${tempnn}.proto
echo "Done"
echo

echo "Initializing new softmax layer randomly"
#Initialize the softmax layer
nnet-initialize --binary=false $nndir/${tempnn}.proto $nndir/${tempnn}.init
echo "Done"
echo

echo "Concatinating two networks"
#Connect the two networks
#Make sure that the oldnn_outdim = softmax_indim dimensions 
nnet-concat ${nnet_stripped} $nndir/${tempnn}.init ${newnn}
echo "New network is stored at" ${newnn}
echo "Done"
echo

##Comment out these lines if there is no need to check work.
#Copy to ASCII for debugging newly created network to binary
#nnet-copy --binary=false $nndir/${newnn}.init $nndir/${newnn}_text.init
#echo "New network is stored at" $nndir/${newnn}_text.init
