#!/bin/bash

export SEED=15270
export PYTORCH_SEED=`expr $SEED / 10`
export NUMPY_SEED=`expr $PYTORCH_SEED / 10`

# path to bert type and path
export BERT_MODEL=bert-base-multilingual-cased
export TOKEN=[SEP]
export MODEL_TYPE=bert

# export BERT_MODEL=roberta-base
# export TOKEN="</s>"
# export MODEL_TYPE=roberta

#path to bert-multilingual
# export BERT_MODEL=bert-base-multilingual-cased
# export TOKEN=[SEP]
# export MODEL_TYPE=bert

# path to distilroberta-polish
# export BERT_MODEL=sdadas/polish-distilroberta
# export TOKEN="</s>"
# export MODEL_TYPE=distilroberta

# path to xlm-roberta
# export BERT_MODEL=xlm-roberta-large
# export TOKEN="</s>"
# export MODEL_TYPE=roberta

# path to dataset files
export TRAIN_PATH=data/ClarinEmo/train.jsonl
export DEV_PATH=data/ClarinEmo/dev.jsonl
export TEST_PATH=data/ClarinEmo/test.jsonl
export TRAINING_DYNAMICS_PATH=data/training_dynamics_tmp

# model
export USE_SEP=true  # true for our model. false for baseline
export WITH_CRF=false  # CRF only works for the baseline

# training params
export cuda_device=0
export BATCH_SIZE=4 # set one for roberta
export LR=1e-5
#export TRAINING_DATA_INSTANCES=1668
export TRAINING_STEPS=52
export NUM_EPOCHS=20

# limit number of sentneces per examples, and number of words per sentence. This is dataset dependant
export MAX_SENT_PER_EXAMPLE=100
export SENT_MAX_LEN=250

# this is for the evaluation of the summarization dataset
export SCI_SUM=false
export USE_ABSTRACT_SCORES=false
export SCI_SUM_FAKE_SCORES=false  # use fake scores for testing

CONFIG_FILE=sequential_sentence_classification/config.jsonnet # for UserID_SSC model use userid_config.jsonnet instead

python3 -m allennlp train $CONFIG_FILE  --include-package sequential_sentence_classification -s $SERIALIZATION_DIR "$@"
