# Recipes
Recipes for training OpenNMT systems


You will find here some "recipes" which basically script the end-to-end data preparation, preprocessing, training and evaluation.

If you clone Recipes.git repo at the same level as OpenNMT.git on your local computer, you don't need to update the PATH
in the scripts. Otherwise update the line OPENNMT_PATH=../../OpenNMT

Baseline-1M-enfr
GPU highly recommended. Training takes 75 minutes per epoch on a single GTX 1080.
Parameters: 2x500 layers, 13 epochs. See script for the details.
Data: set of 1 million parallel sentences (extract of Europarl, Newscommentaries, ..)
See the results file for the evaluation.

