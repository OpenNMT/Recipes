#!/bin/bash
#
# Copyright 2017 Ubiqus   (Author: Vincent Nguyen)
#		 Systran  (Author: Jean Senellart)
# License MIT
#
# This recipe shows how to build an openNMT translation model from English to French
# based on a limited resource (1 Mio segments)
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

# making these variables to make replication easier for other languages
sl=en
tl=fr

# training corpus - baseline-1M/baseline-2M available
corpus=baseline-1M

# At the moment only "stage" option is available anyway
. local/parse_options.sh

# Data download and preparation

if [ $stage -le 0 ]; then
# TODO put this part in a local/download_data.sh script ?
  mkdir -p data
  cd data
  if [ ! -f $corpus-$sl$tl.tgz ]; then
    echo "$0: downloading the baseline corpus from amazon s3"
    wget https://s3.amazonaws.com/opennmt-trainingdata/$corpus-$sl$tl.tgz
    tar xzfv $corpus-$sl$tl.tgz
  fi
  if [ ! -f testsets-$sl$tl.tgz ]; then
    echo "$0: downloading the baseline corpus from amazon s3"
    wget https://s3.amazonaws.com/opennmt-tests/testsets-$sl$tl.tgz
    tar xzfv testsets-$sl$tl.tgz
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
  cd ..
fi

# Tokenize the Corpus
if [ $stage -le 1 ]; then
  echo "$0: tokenizing corpus and test sets"
  for f in data/$corpus-$sl$tl/*.?? ; do th tools/tokenize.lua -case_feature -joiner_annotate < $f > $f.tok ; done
fi

# Preprocess the data - decide here the vocabulary size 50000 default value
if [ $stage -le 2 ]; then
  mkdir -p exp
  echo "$0: preprocessing corpus"
  th preprocess.lua -src_vocab_size 50000 -tgt_vocab_size 50000 \
  -train_src data/$corpus-$sl$tl/*_train.$sl.tok \
  -train_tgt data/$corpus-$sl$tl/*_train.$tl.tok \
  -valid_src data/$corpus-$sl$tl/*_valid.$sl.tok \
  -valid_tgt data/$corpus-$sl$tl/*_valid.$tl.tok -save_data exp/data-$corpus-$sl$tl
fi

# Train the model !!!! even if OS cuda device ID is 0 you need -gpuid=1
# Decide here the number of epochs, learning rate, which epoch to start decay, decay rate
# if you change number of epochs do not forget to change the model name too
if [ $stage -le 3 ]; then
  if [ $notrain = false ]; then
    echo "$0: training starting, will take a while."
    th train.lua -data  exp/data-$corpus-$sl$tl-train.t7 \
    -save_model exp/model-$corpus-$sl$tl \
    -end_epoch 13 -start_decay_at 5 -learning_rate_decay 0.65 -gpuid 1
    cp -f exp/model-$corpus-$sl$tl"_epoch13_"*".t7" exp/model-$corpus-$sl$tl"_final.t7"
  else
    echo "$0: using an existing model"
    if [ ! -f exp/model-$corpus-$sl$tl"_final.t7" ]; then
      echo "$0: mode file does not exist"
      exit 1
    fi
  fi
fi

# Deploy model for CPU usage
if [ $stage -le 4 ]; then
  if [ $decode_cpu = true ]; then
    th tools/release_model.lua -force -model exp/model-$corpus-$sl$tl"_final.t7" -output_model exp/model-$corpus-$sl$tl"_cpu.t7" -gpuid 1
  fi
fi

# Translate using gpu
# you can change this by changing the model name from _final to _cpu and remove -gpuid 1
if [ $stage -le 5 ]; then
  [ $decode_cpu = true ] && dec_opt="" || dec_opt="-gpuid 1"
  th translate.lua -replace_unk -model exp/model-$corpus-$sl$tl"_final.t7" \
  -src data/$corpus-$sl$tl/*_test.$sl.tok -output exp/${corpus}_test.hyp.$tl.tok $dec_opt
fi

# Evaluate the generic test set with multi-bleu
if [ $stage -le 6 ]; then
  th tools/detokenize.lua -case_feature < exp/${corpus}_test.hyp.$tl.tok > exp/${corpus}_test.hyp.$tl.detok
  perl local/multi-bleu.perl data/$corpus-$sl$tl/*_test.$tl \
  < exp/${corpus}_test.hyp.$tl.detok > exp/${corpus}_test_multibleu.txt
fi

###############################
#### Newstest Evaluation
####

if [ $stage -le 7 ]; then

testset=newstest2014-$sl$tl

  perl local/input-from-sgm.perl < data/testsets-$sl$tl/News/$testset-src.$sl.sgm \
  > data/testsets-$sl$tl/News/$testset-src.$sl

  th tools/tokenize.lua -case_feature -joiner_annotate < data/testsets-$sl$tl/News/$testset-src.$sl \
  > data/testsets-$sl$tl/News/$testset-src.$sl.tok

  [ $decode_cpu = true ] && dec_opt="" || dec_opt="-gpuid 1"

  th translate.lua -replace_unk -model exp/model-$corpus-$sl$tl"_final"*.t7 \
  -src data/testsets-$sl$tl/News/$testset-src.$sl.tok \
  -output exp/$testset-tgt.trans.$tl.tok $dec_opt

  th tools/detokenize.lua -case_feature < exp/$testset-tgt.trans.$tl.tok \
  > exp/$testset-tgt.trans.$tl

# Wrap-xml to convert to sgm
  perl local/wrap-xml.perl $tl data/testsets-$sl$tl/News/$testset-src.$sl.sgm tst \
  < exp/$testset-tgt.trans.$tl \
  > exp/$testset-tgt.trans.$tl.sgm

  perl local/mteval-v13a.pl -r data/testsets-$sl$tl/News/$testset-ref.$tl.sgm \
  -s data/testsets-$sl$tl/News/$testset-src.$sl.sgm -t exp/$testset-tgt.trans.$tl.sgm \
  -c > exp/nist-bleu-$testset
fi

