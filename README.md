# Recipes
Recipes for training OpenNMT systems


You will find here some "recipes" which basically script the end-to-end data preparation, preprocessing, training and evaluation.

## Requirements

* You do need OpenNMT - see [here](http://opennmt.net/OpenNMT/installation/). If you clone Recipes.git repo at the same level as OpenNMT.git on your local computer, you don't need to update the PATH
in the scripts. Otherwise update the line `OPENNMT_PATH=../../OpenNMT`
* for evaluation scripts, you do need perl `XML::Twig` module (`perl -MCPAN -e 'install XML::Twig`)

## The recipes

### Baseline-1M-enfr
Train a baseline English-French model, use case feature and onmt reversible tokenization.  GPU highly recommended. Training takes 75 minutes per epoch on a single GTX 1080.
Parameters: 2x500 layers, 13 epochs. See script for the details.
Data: set of 1 million parallel sentences (extract of Europarl, Newscommentaries, ..)
See the results file for the evaluation.

### Romance Multi-way
See http://forum.opennmt.net/t/training-romance-multi-way-model/86  
GPU highly recommended. Training takes 4 1/2 hours per epoch on a single GTX 1080.
Parameters: 2x500 layers, 13 epochs. See script for the details.
