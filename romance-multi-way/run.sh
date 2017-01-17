#!/bin/bash
#
# Copyright 2017 Ubiqus   (Author: Vincent Nguyen)
#		 Systran  (Author: Jean Senellart)
# License MIT
#
# This recipe shows how to build an openNMT translation model for Romance Multi way languages
# based on 200 000 parallel sentences for each pair
#
# Based on the tuto from the OpenNMT forum


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

# Data download and preparation

if [ $stage -le 0 ]; then
# TODO put this part in a local/download_data.sh script ?
  mkdir -p data
  cd data
  if [ ! -f multi-esfritptro-parallel.tgz ]; then
    echo "$0: downloading the baseline corpus from amazon s3"
    wget https://s3.amazonaws.com/opennmt-trainingdata/multi-esfritptro-parallel.tgz
    tar xzfv multi-esfritptro-parallel.tgz
  fi
  cd ../local
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

# Tokenize and prepare the Corpus
if [ $stage -le 1 ]; then
  echo "$0: tokenizing corpus"
  for f in data/train*.?? ; do th tools/tokenize.lua < $f > $f.rawtok ; done
  cat data/train*.rawtok | python local/learn_bpe.py -s 32000 > data/esfritptro.bpe32000
  for f in data/*-????.?? ; do \
    th tools/tokenize.lua -case_feature -joiner_annotate -nparrallel 4 -bpe_model data/esfritptro.bpe32000 < $f > $f.tok
  done
  for set in train valid test ; do rm data/$set-multi.???.tok ; done
  for src in es fr it pt ro ; do
    for tgt in es fr it pt ro ; do
      [ ! $src = $tgt ] && perl -i.bak -pe "s//__opt_tgt_$tgt\xEF\xBF\xA8N /" data/*-$src$tgt.$src.tok
      for set in train valid test ; do
        [ ! $src = $tgt ] && cat data/$set-$src$tgt.$src.tok >> data/$set-multi.src.tok
        [ ! $src = $tgt ] && cat data/$set-$src$tgt.$tgt.tok >> data/$set-multi.tgt.tok
      done
    done
  done
  paste data/valid-multi.src.tok data/valid-multi.tgt.tok | shuf > data/valid-multi.srctgt.tok
  head -2000 data/valid-multi.srctgt.tok | cut -f1 > data/valid-multi2000.src.tok
  head -2000 data/valid-multi.srctgt.tok | cut -f2 > data/valid-multi2000.tgt.tok
fi

# Preprocess the data - decide here the vocabulary size 50000 default value
if [ $stage -le 2 ]; then
  mkdir -p exp
  echo "$0: preprocessing corpus"
  th preprocess.lua -src_vocab_size 50000 -tgt_vocab_size 50000 \
  -train_src data/train-multi.src.tok -train_tgt data/train-multi.tgt.tok \
  -valid_src data/valid-multi2000.src.tok -valid_tgt data/valid-multi2000.tgt.tok \
  -save_data exp/model-multi
fi

# Train the model !!!! even if OS cuda device ID is 0 you need -gpuid=1
# Decide here the number of epochs, learning rate, which epoch to start decay, decay rate
# if you change number of epochs do not forget to change the model name too
# This example has a smaller topology compared to tuto for faster training (worse results)
if [ $stage -le 3 ]; then
  if [ $notrain = false ]; then
    echo "$0: training starting, will take a while."
    th train.lua -layers 2 -rnn_size 500 -brnn -word_vec_size 600 \
    -end_epoch 13 -learning_rate 1 -start_decay_at 5 -learning_rate_decay 0.65 \
    -data  exp/model-multi-train.t7 -save_model exp/model-multi-2-500-600 -gpuid 1
    cp -f exp/model-multi-2-500-600"_epoch13_"*".t7" exp/model-multi-2-500-600"_final.t7"
  else
    echo "$0: using an existing model"
    if [ ! -f exp/model-multi-2-500-600"_final.t7" ]; then
      echo "$0: mode file does not exist"
      exit 1
    fi
  fi
fi

# Deploy model for CPU usage
if [ $stage -le 4 ]; then
  if [ $decode_cpu = true ]; then
    th tools/release_model.lua -force -model exp/model-multi-2-500-600"_final.t7" \
    -output_model exp/model-multi-2-500-600"_cpu.t7" -gpuid 1
  fi
fi

# Translate using gpu
# you can change this by changing the model name from _final to _cpu and remove -gpuid 1
if [ $stage -le 5 ]; then
  [ $decode_cpu = true ] && dec_opt="" || dec_opt="-gpuid 1"
  for src in es fr it pt ro ; do
    for tgt in es fr it pt ro ; do
      [ ! $src = $tgt ] && th translate.lua -replace_unk -model exp/model-multi-2-500-600"_final"*".t7" \
       -src data/test-$src$tgt.$src.tok -output exp/test-$src$tgt.hyp.$tgt.tok $dec_opt
    done
  done
fi

# Evaluate the generic test set with multi-bleu
if [ $stage -le 6 ]; then
  for src in es fr it pt ro ; do
    for tgt in es fr it pt ro ; do
      [ ! $src = $tgt ] && th tools/detokenize.lua -case_feature < exp/test-$src$tgt.hyp.$tgt.tok \
      > exp/test-$src$tgt.hyp.$tgt.detok
      [ ! $src = $tgt ] && local/multi-bleu.perl data/test-$src$tgt.$tgt \
      < exp/test-$src$tgt.hyp.$tgt.detok > exp/test-$src$tgt"_multibleu".txt
    done
  done
fi


