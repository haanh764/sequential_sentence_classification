import json
import shutil
import sys

from allennlp.commands import main
import os

os.environ["SEED"] = "15270"
os.environ["PYTORCH_SEED"] = "1527"
os.environ["NUMPY_SEED"] = "1527"


# path to bert vocab and weights
os.environ["BERT_MODEL"] = "allenai/scibert_scivocab_uncased"
#os.environ["BERT_MODEL"] = "roberta-base"
#os.environ["BERT_MODEL"] = "camembert-base"


#os.environ["TOKEN"] = "</s>"
#os.environ["MODEL_TYPE"] = "roberta"

os.environ["TOKEN"] = "[SEP]"
os.environ["MODEL_TYPE"] = "bert"

# path to dataset files
os.environ["TRAIN_PATH"] = "data/CSAbstruct/train.jsonl"
os.environ["DEV_PATH"] = "data/CSAbstruct/dev.jsonl"
os.environ["TEST_PATH"] = "data/CSAbstruct/test.jsonl"

# path to dataset files
# os.environ["TRAIN_PATH"] = "data_copy/CSAbstruct/train.jsonl"
# os.environ["DEV_PATH"] = "data_copy/CSAbstruct/dev.jsonl"
# os.environ["TEST_PATH"] = "data_copy/CSAbstruct/test.jsonl"


# path to dataset files CC
# os.environ["TRAIN_PATH"] = "data/zonage-train/train.jsonl"
# os.environ["DEV_PATH"] = "data/zonage-train/dev.jsonl"
# os.environ["TEST_PATH"] = "data/zonage-train/test.jsonl"

# model
os.environ["USE_SEP"] = "false"
os.environ["WITH_CRF"] = "true"

# training params
os.environ["cuda_device"] = "0"
os.environ["BATCH_SIZE"] = "2"
os.environ["LR"] = "1e-5"
os.environ["STEPS_PER_EPOCH"] = "55"

os.environ["NUM_EPOCHS"] = "30"

os.environ["SENT_MAX_LEN"] = "300"

# this is for the evaluation of the summarization dataset
os.environ["SCI_SUM"] = "false"
os.environ["USE_ABSTRACT_SCORES"] = "false"
os.environ["SCI_SUM_FAKE_SCORES"] = "false"

os.environ["METRIC"] = "+avgF"

config_file = "sequential_sentence_classification/config.jsonnet"

# Use overrides to train on CPU.
overrides = json.dumps({"trainer": {"cuda_device": 0}})

serialization_dir = "/tmp/debugger_train"

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
shutil.rmtree(serialization_dir, ignore_errors=True)

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "sequential_sentence_classification",
    ]

main()