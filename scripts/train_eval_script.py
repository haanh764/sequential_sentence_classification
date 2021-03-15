import json
import shutil
import sys
import yaml

from allennlp.commands import main
import os

params_file = "./params/params_train.yaml"
metrics_file = "./metrics/metrics_eval_test.json"
config_file = "sequential_sentence_classification/config.jsonnet"

# this is for the evaluation of the summarization dataset
os.environ["SCI_SUM"] = "false"
os.environ["USE_ABSTRACT_SCORES"] = "false"
os.environ["SCI_SUM_FAKE_SCORES"] = "false"
###

# import params
with open(params_file, 'r') as fd:
    params = yaml.safe_load(fd)

serialization_dir = "outputs/"+ params["model"]["name"]
model_path = serialization_dir + "/model.tar.gz"
test_file = params["reader"]["test_path"]

os.environ["cuda_device"] = params["global"]["cuda"]

os.environ["SEED"] = params["global"]["seed"] 
os.environ["PYTORCH_SEED"] = str(int(params["global"]["seed"]) / 10)
os.environ["NUMPY_SEED"] = str(int(params["global"]["seed"]) / 10)

# model
os.environ["USE_SEP"] = params["global"]["use_sep"]
os.environ["WITH_CRF"] = params["global"]["with_crf"]

# path to bert vocab and weights
os.environ["BERT_MODEL"] = params["reader"]["bert_model"]
os.environ["TOKEN"] = params["reader"]["token"]
os.environ["MODEL_TYPE"] = params["reader"]["model_type"]

# path to dataset files
os.environ["TRAIN_PATH"] = params["reader"]["train_path"]
os.environ["DEV_PATH"] = params["reader"]["test_path"]
os.environ["TEST_PATH"] = params["reader"]["dev_path"]

# training params

os.environ["BATCH_SIZE"] = params["train"]["batch_size"]
os.environ["TRAIN_PARAMS"] = params["train"]["train_parameters"]
os.environ["LAST_LAYER_ONLY"] = params["train"]["last_layer_only"]
os.environ["SENT_MAX_LEN"] = params["train"]["sent_max_len"]

# trainer
os.environ["METRIC"] = params["trainer"]["metric"]
os.environ["PATIENCE"] = params["trainer"]["patience"]
os.environ["ACCUM_STEPS"] = params["trainer"]["accumulation_steps"]
os.environ["LR"] = params["trainer"]["lr"]
os.environ["WEIGHT_DECAY"] = params["trainer"]["weight_decay"]
os.environ["NUM_EPOCHS"] = params["trainer"]["n_epochs"]
os.environ["STEPS_PER_EPOCH"] = params["trainer"]["steps_per_epoch"]

### attention
os.environ["INPUT_DIM"] = params["attention"]["input_dim"]
os.environ["HIDDEN_DIM"] = params["attention"]["feedforward_hidden_dim"]
os.environ["NUM_LAYERS"] = params["attention"]["num_layers"]
os.environ["NUM_ATT_HEADS"] = params["attention"]["num_attention_heads"]

# Assemble the command into sys.argv
a = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "sequential_sentence_classification",
    ]

b = [
    "allennlp",  # command name, not used by main
    "evaluate",
    model_path,
    test_file,
    "--include-package", "sequential_sentence_classification",
    "--output-file", metrics_file,
    ]
main(a)
main(b)