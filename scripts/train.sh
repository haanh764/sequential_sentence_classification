#!/bin/bash

export SEED=15270
export PYTORCH_SEED=`expr $SEED / 10`
export NUMPY_SEED=`expr $PYTORCH_SEED / 10`

# path to bert type and path
export BERT_MODEL=allenai/scibert_scivocab_uncased


# export BERT_MODEL=roberta-base


# path to dataset files
export TRAIN_PATH=data/CSAbstruct/train.jsonl
export DEV_PATH=data/CSAbstruct/dev.jsonl
export TEST_PATH=data/CSAbstruct/test.jsonl

# model
export WITH_CRF=false  # CRF only works for the baseline

# training params
export cuda_device=0
export BATCH_SIZE=4 # set one for roberta
export LR=1e-5
#export TRAINING_DATA_INSTANCES=1668
export TRAINING_STEPS=52
export NUM_EPOCHS=20

# limit number of sentneces per examples, and number of words per sentence. This is dataset dependant
export MAX_SENT_PER_EXAMPLE=10
export SENT_MAX_LEN=80

CONFIG_FILE=sequential_sentence_classification/config.jsonnet

python3 -m allennlp train $CONFIG_FILE  --include-package sequential_sentence_classification -s $SERIALIZATION_DIR "$@"
