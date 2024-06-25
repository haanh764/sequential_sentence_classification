# <p align=center>Sequential Sentence Classification methods in emotion recognition</p>

This repo contain code for our methods for emotion recognition based on sequential sentence classification with explainable data cartography approach

### How to run

```
pip install -r requirements.txt
scripts/train.sh tmp_output_dir
```

- `model_multiclass_ssc.py`: model based on the original sequential sentence classification model
- `model_multilabel_ssc.py`: model for multilabel classification based on sequential sentence classification model

Update the `scripts/train.sh` script with the appropriate hyperparameters and datapaths.

### Dataset

Split and put train, dev, test splits of the dataset in `data/ClarinEmo`

### Citing

If you use the data or the model, please cite,
```
@inproceedings{Cohan2019EMNLP,
  title={Pretrained Language Models for Sequential Sentence Classification},
  author={Arman Cohan, Iz Beltagy, Daniel King, Bhavana Dalvi, Dan Weld},
  year={2019},
  booktitle={EMNLP},
}
```