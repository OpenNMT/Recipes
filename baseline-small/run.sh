#!/bin/bash
#
# Copyright 2017 Ubiqus   (Author: Vincent Nguyen)
#		 
# License MIT
#
# This recipe shows how to build an openNMT translation model from French to English
# based on 
# Global Voices
# News Commentary v11
# This script does not download the datasets, you need to drop the files in data/public
# same for the test set newstest2014
# making these variables to make replication easier for other languages

sl=fr
tl=en
corpus[1]=data/public/News-Commentary11.en-fr.clean
corpus[2]=data/public/GlobalVoices.en-fr.clean

vocab_size=50000
seq_len=50

testset=newstest2014-fren

use_bpe=false
bpe_size=32000
[ $use_bpe = false ] && bpe_model="" || bpe_model="-bpe_model data/train-$sl$tl.bpe32000"
use_case=false
[ $use_case = false ] && case_feat="" || case_feat="-case_feature"

# TODO test is GPU is present or not
CUDA_VISIBLE_DEVICES=0
decode_cpu=false

# Make symlinks to access OpenNMT scripts - change this line if needed
OPENNMT_PATH=../../OpenNMT
[ ! -h tools ] && ln -s $OPENNMT_PATH/tools tools
[ ! -h preprocess.lua ] && ln -s $OPENNMT_PATH/preprocess.lua preprocess.lua
[ ! -h train.lua ] && ln -s $OPENNMT_PATH/train.lua train.lua
[ ! -h translate.lua ] && ln -s $OPENNMT_PATH/translate.lua translate.lua
[ ! -h onmt ] && ln -s $OPENNMT_PATH/onmt onmt

# this is usefull to skip some stages during step by step execution
stage=0

# if you want to run without training and use an existing model in the "exp" folder set notrain to true
notrain=false

# At the moment only "stage" option is available anyway
. local/parse_options.sh

function score_epoch {
  # convert sgm input into text file
  local/input-from-sgm.perl < data/public/test/$testset-src.$sl.sgm > data/$testset-src.$sl
  # tokenize the text file
  th tools/tokenize.lua $case_feat -mode aggressive -joiner_annotate $bpe_model < data/$testset-src.$sl > data/$testset-src.$sl.tok
  # translate the test set
  [ $decode_cpu = true ] && dec_opt="" || dec_opt="-gpuid 1"
  th translate.lua -replace_unk -disable_logs -model exp/model-$sl$tl"_epoch"$1"_"*.t7 \
  -src data/$testset-src.$sl.tok \
  -output exp/$testset-tgt.trans.$tl.tok $dec_opt
  # detokenize
  th tools/detokenize.lua $case_feat < exp/$testset-tgt.trans.$tl.tok \
  > exp/$testset-tgt.trans.$tl
  # Wrap-xml to convert to sgm the translated text
  local/wrap-xml.perl $tl data/public/test/$testset-src.$sl.sgm tst \
  < exp/$testset-tgt.trans.$tl > exp/$testset-tgt.trans.$tl.sgm
  # compute the bleu score
  local/mteval-v13a.pl -r data/public/test/$testset-ref.$tl.sgm \
  -s data/public/test/$testset-src.$sl.sgm -t exp/$testset-tgt.trans.$tl.sgm \
  -c > exp/nist-bleu-$testset-epoch-$1

  [ $decode_cpu = true ] && dec_opt="" || dec_opt="-gpuid 1"
  th translate.lua -replace_unk -disable_logs -model exp/model-$sl$tl"_epoch"$1"_"*".t7" \
  -src data/valid.$sl.tok -output exp/valid.hyp.$tl.tok $dec_opt

  th tools/detokenize.lua $case_feat < exp/valid.hyp.$tl.tok > exp/valid.hyp.$tl.detok
  th tools/detokenize.lua $case_feat < data/valid.$tl.tok > exp/valid.$tl.detok
  local/multi-bleu.perl exp/valid.$tl.detok \
  < exp/valid.hyp.$tl.detok > exp/generic_test_multibleu-detok-epoch$1.txt
  local/multi-bleu.perl data/valid.$tl.tok \
  < exp/valid.hyp.$tl.tok > exp/generic_test_multibleu-tok-epoch$1.txt
}


if [ $stage -le 0 ]; then
  cd local
  if [ ! -f mteval-v13a.pl ]; then
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/mteval-v13a.pl
  fi
  if [ ! -f input-from-sgm.perl ]; then
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/ems/support/input-from-sgm.perl
  fi
  if [ ! -f wrap-xml.perl ]; then
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/ems/support/wrap-xml.perl
  fi
  if [ ! -f multi-bleu.perl ]; then
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl
  fi
  if [ ! -f learn_bpe.py ]; then
    wget https://raw.githubusercontent.com/rsennrich/subword-nmt/master/learn_bpe.py
  fi
  cd ..
fi


# Prepare Corpus, build BPE model, build dictionary
if [ $stage -le 1 ]; then

  if $use_bpe; then
    echo "$0: tokenizing corpus for BPE modelling"
    for ((i=1; i<= ${#corpus[@]}; i++))
    do
      for f in ${corpus[$i]}.$sl ${corpus[$i]}.$tl 
      do 
       file=$(basename $f)
       th tools/tokenize.lua -mode aggressive -nparallel 6 < $f > data/$file.rawtok
      done
    done
    cat data/*.rawtok | python local/learn_bpe.py -s $bpe_size > data/train-$sl$tl.bpe$bpe_size
    rm data/*.rawtok
  fi

  echo "$0: tokenizing corpus"
  for ((i=1; i<= ${#corpus[@]}; i++))
  do
    for f in ${corpus[$i]}.$sl ${corpus[$i]}.$tl 
    do 
     file=$(basename $f)
     th tools/tokenize.lua -mode aggressive $case_feat -joiner_annotate -nparallel 6 \
     $bpe_model < $f > data/$file.tok
    done
  done

  echo "$0: building dictionaries based on public and private data"
  cat data/*.$sl.tok > data/tempo.$sl.tok
  cat data/*.$tl.tok > data/tempo.$tl.tok
  th tools/build_vocab.lua -data data/tempo.$sl.tok -save_vocab data/dict.$sl -vocab_size $vocab_size
  th tools/build_vocab.lua -data data/tempo.$tl.tok -save_vocab data/dict.$tl -vocab_size $vocab_size
  rm data/tempo.??.tok

  echo "$0: preparing public and private training sets"
  for ((i=1; i<= ${#corpus[@]}; i++))
  do
    file=$(basename ${corpus[$i]}.$sl)
    cat data/$file.tok >> data/train-full.$sl.tok
    file=$(basename ${corpus[$i]}.$tl)
    cat data/$file.tok >> data/train-full.$tl.tok
  done

  local/testset.pl -n 2000 -o data/valid.$sl.tok -h data/train.$sl.tok < data/train-full.$sl.tok > lines-tmp.txt
  local/lineextract.pl lines-tmp.txt < data/train-full.$tl.tok > data/valid.$tl.tok
  local/heldextract.pl lines-tmp.txt < data/train-full.$tl.tok > data/train.$tl.tok
  rm data/train-full.*.tok
  rm lines-tmp.txt

fi


# Preprocess the data - decide here the vocabulary size 50000 default value
if [ $stage -le 2 ]; then
  mkdir -p exp
  echo "$0: preprocessing corpus"
    th preprocess.lua -src_vocab_size $vocab_size -tgt_vocab_size $vocab_size \
    -src_seq_length $seq_len -tgt_seq_length $seq_len \
    -train_src data/train.$sl.tok -train_tgt data/train.$tl.tok \
    -valid_src data/valid.$sl.tok -valid_tgt data/valid.$tl.tok \
    -src_vocab data/dict.$sl.dict -tgt_vocab data/dict.$tl.dict \
    -save_data exp/data-$sl$tl
fi

# Train the model !!!! even if OS cuda device ID is 0 you need -gpuid=1
# Decide here the number of epochs, learning rate, which epoch to start decay, decay rate
# if you change number of epochs do not forget to change the model name too

# Train on corpus

if [ $stage -le 3 ]; then
    learning_rate=1
    start_decay_at=6
    learning_rate_decay=0.5
    echo "$0: training public corpus starting, will take a while."
      # train first epoch
      th train.lua -layers 2 -rnn_size 512 -data exp/data-$sl$tl-train.t7 \
      -save_model exp/model-$sl$tl -dropout 0.3 -report_every 500 -word_vec_size 512 \
      -start_epoch 1 -end_epoch 1 -max_batch_size 32 \
      -learning_rate $learning_rate -start_decay_at $start_decay_at \
      -learning_rate_decay $learning_rate_decay -gpuid 1
      # score it -sample 50000 -sample_tgt_vocab -sample_type partition 
      score_epoch 1
 #     th tools/release_model.lua -force -model exp/model-$sl$tl"_epoch1_"*".t7" \
 #      -output_model exp/modelcpu-$sl$tl"_epoch1.t7" -gpuid 1

      for epoch in 2 3 4 5 6 7 8 9 10
      do
        prev_epoch=$(expr $epoch - 1)
        [ $epoch -ge $start_decay_at ] && \
        learning_rate=`awk 'BEGIN{printf("%0.4f", '$learning_rate' * '$learning_rate_decay')}'`
        th train.lua -rnn_size 512 -train_from exp/model-$sl$tl"_epoch"$prev_epoch"_"*".t7" \
        -data exp/data-$sl$tl-train.t7 \
        -save_model exp/model-$sl$tl -report_every 500 -word_vec_size 512 \
        -start_epoch $epoch -end_epoch $epoch -max_batch_size 32 \
        -learning_rate $learning_rate -start_decay_at $start_decay_at \
        -learning_rate_decay $learning_rate_decay -gpuid 1
        # score it
        score_epoch $epoch
#        th tools/release_model.lua -force -model exp/model-$sl$tl"_epoch"$epoch"_"*".t7" \
#         -output_model exp/modelcpu-$sl$tl"_epoch"$epoch".t7" -gpuid 1
      done
fi
