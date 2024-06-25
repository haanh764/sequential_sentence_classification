import logging
from typing import Dict, List, Optional, Any
import os
import json

import numpy as np
import torch
from torch.nn import Linear
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.conditional_random_field import ConditionalRandomField
from allennlp.training.metrics import Metric
from allennlp.training.callbacks.callback import TrainerCallback
from allennlp.training.trainer import Trainer
from torchmetrics.classification import MulticlassF1Score

logger = logging.getLogger(__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@TrainerCallback.register("SaveTrainingDynamics")
class SaveTrainingDynamics(TrainerCallback):
    def __init__(self, output_dir: str = None) -> None:
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
    def on_epoch(self, trainer: Trainer, metrics: Dict[str, Any], **kwargs) -> None:
        dynamics = []
        model = trainer.model

        model.eval()
        with torch.no_grad():
            for batch in trainer.data_loader:
                output = model.forward(**batch)
                logits = output["logits"].detach().cpu().numpy()
                golds = output["golds"].detach().cpu().numpy()
                guids = batch['metadata'][0]

                assert len(guids) == len(golds)
                for gold, logit, guid in zip(golds, logits, guids):
                    data = {
                        "guid": guid['guid'],
                        "gold": int(gold),
                        "logits": logit.tolist()
                    }
                    dynamics.append(data)

        # Write the dynamics to a JSONL file
        epoch_number = trainer._epochs_completed
        print(epoch_number)
        
        filename = f"dynamics_epoch_{epoch_number}.jsonl"
        output_path = os.path.join(self.output_dir, filename)

        with open(output_path, "w") as f:
          f.write(json.dumps(dynamics) + "\n")

class F1Score(Metric):
    def __init__(self, num_classes, threshold=0.5, average='samples', label_names=None):
        self.num_labels = num_classes
        self.threshold = threshold
        self.f1_score = MulticlassF1Score(num_classes=num_classes, average=average, threshold=threshold).to(device)
        self._f1 = 0.0
        self._count = 0
        self.label_names = label_names

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: Optional[torch.Tensor] = None):
        self.f1_score(predictions, gold_labels)

    def get_metric(self, reset: bool = False):        
        if reset:
            self.reset()

        if self.label_names:
            f1_scores = {}
            for i, name in enumerate(self.label_names):
                f1_scores[name] = self.f1_score.compute()[i].item()
            return f1_scores
        else:
            f1_score = self.f1_score.compute().item()
            return {'f1_score': f1_score}

    def reset(self):
        self.f1_score.reset()
        self._f1 = 0.0
        self._count = 0

@Model.register("SeqClassificationModel")
class SeqClassificationModel(Model):
    """
    Question answering model where answers are sentences
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 use_sep: bool = True,
                 with_crf: bool = False,
                 self_attn: Seq2SeqEncoder = None,
                 bert_dropout: float = 0.1,
                 sci_sum: bool = False,
                 intersentence_token: str = "[SEP]",
                 model_type: str = "bert",
                 additional_feature_size: int = 0,
                 ) -> None:
        super(SeqClassificationModel, self).__init__(vocab)

        self.text_field_embedder = text_field_embedder
        self.vocab = vocab
        self.use_sep = use_sep
        self.with_crf = with_crf
        self.sci_sum = sci_sum
        self.self_attn = self_attn
        self.additional_feature_size = additional_feature_size
        self.token = intersentence_token
        self.model_type = model_type
        self.dropout = torch.nn.Dropout(p=bert_dropout)

       # define loss
        if self.sci_sum:
            self.loss = torch.nn.MSELoss(reduction='none')  # labels are rouge scores
            self.labels_are_scores = True
            self.num_labels = 1
        else:
            self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            self.labels_are_scores = False
            self.num_labels = self.vocab.get_vocab_size(namespace='labels')
            # define accuracy metrics
            self.label_accuracy = CategoricalAccuracy()
            self.label_f1_metrics_macro = {}
            self.label_f1_metrics_micro = {}

            # define F1 metrics per label
            label_names = []
            for label_index in range(self.num_labels):
                label_name = self.vocab.get_token_from_index(namespace='labels', index=label_index)
                label_names.append(label_name)

            self.label_f1_metrics_macro = F1Score(num_classes=self.vocab.get_vocab_size(namespace='labels'), average="macro")
            self.label_f1_metrics_micro = F1Score(num_classes=self.vocab.get_vocab_size(namespace='labels'), average="micro")
            self.label_f1_metric = F1Score(num_classes=self.vocab.get_vocab_size(namespace='labels'), average=None, label_names=label_names)

        encoded_senetence_dim = text_field_embedder._token_embedders['bert'].get_output_dim()

        ff_in_dim = encoded_senetence_dim if self.use_sep else self_attn.get_output_dim()
        ff_in_dim += self.additional_feature_size

        self.time_distributed_aggregate_feedforward = Linear(ff_in_dim, self.num_labels)

        if self.with_crf:
            self.crf = ConditionalRandomField(
                self.num_labels, constraints=None,
                include_start_end_transitions=True
            )

    def forward(self,  # type: ignore
                sentences: torch.LongTensor,
                labels: torch.IntTensor = None,
                confidences: torch.Tensor = None,
                additional_features: torch.Tensor = None,
                metadata: List[Dict[str, Any]] = None,
                ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        TODO: add description

        """
        # ===========================================================================================================
        # Input: sentences
        # Output: embedded_sentences

        # embedded_sentences: batch_size, num_sentences, sentence_length, embedding_size
        embedded_sentences = self.text_field_embedder(sentences, num_wrapping_dims= 1)
        mask = get_text_field_mask(sentences, num_wrapping_dims=1).float()
        batch_size, num_sentences, _, _ = embedded_sentences.size()

        if self.use_sep:
            # The following code collects vectors of the SEP tokens from all the examples in the batch,
            # and arrange them in one list. It does the same for the labels and confidences.
            # TODO: replace 103 with '[SEP]'
            index_sep = int(self.vocab.get_token_index(token=self.token, namespace = "tags"))
            sentences_mask = sentences['bert']["token_ids"] == index_sep # mask for all the SEP tokens in the batch
            embedded_sentences = embedded_sentences[sentences_mask]  # given batch_size x num_sentences_per_example x sent_len x vector_len
                                                                        # returns num_sentences_per_batch x vector_len
            ## roberta only WORKS ONLY IF BATCH SIZE == 1
            if (self.model_type == "roberta") or (self.model_type == "distilroberta"):            
                assert batch_size == 1, "set batch size to 1 for RoBERTa"                                               
                indx = np.arange(embedded_sentences.shape[0])
                device = "cuda" 
                sel_idx = torch.from_numpy(indx[indx%2==0]).to(device)# select only scond intersentence marker
                embedded_sentences = torch.index_select(embedded_sentences, 0, sel_idx)
            
            assert embedded_sentences.dim() == 2
            num_sentences = embedded_sentences.shape[0]
            # for the rest of the code in this model to work, think of the data we have as one example
            # with so many sentences and a batch of size 1
            batch_size = 1
            embedded_sentences = embedded_sentences.unsqueeze(dim=0)
            embedded_sentences = self.dropout(embedded_sentences)

            if labels is not None:
                if self.labels_are_scores:
                    labels_mask = labels != 0.0  # mask for all the labels in the batch (no padding)
                else:
                    labels_mask = labels != -1  # mask for all the labels in the batch (no padding)

                labels = labels[labels_mask]  # given batch_size x num_sentences_per_example return num_sentences_per_batch
                assert labels.dim() == 1
                if confidences is not None:
                    confidences = confidences[labels_mask]
                    assert confidences.dim() == 1
                if additional_features is not None:
                    additional_features = additional_features[labels_mask]
                    assert additional_features.dim() == 2

                num_labels = labels.shape[0]
                if num_labels != num_sentences:  # bert truncates long sentences, so some of the SEP tokens might be gone
                    assert num_labels > num_sentences  # but `num_labels` should be at least greater than `num_sentences`
                    logger.warning(f'Found {num_labels} labels but {num_sentences} sentences')
                    labels = labels[:num_sentences]  # Ignore some labels. This is ok for training but bad for testing.
                                                        # We are ignoring this problem for now.
                                                        # TODO: fix, at least for testing

                # do the same for `confidences`
                if confidences is not None:
                    num_confidences = confidences.shape[0]
                    if num_confidences != num_sentences:
                        assert num_confidences > num_sentences
                        confidences = confidences[:num_sentences]

                # and for `additional_features`
                if additional_features is not None:
                    num_additional_features = additional_features.shape[0]
                    if num_additional_features != num_sentences:
                        assert num_additional_features > num_sentences
                        additional_features = additional_features[:num_sentences]

                # similar to `embedded_sentences`, add an additional dimension that corresponds to batch_size=1
                labels = labels.unsqueeze(dim=0)
                if confidences is not None:
                    confidences = confidences.unsqueeze(dim=0)
                if additional_features is not None:
                    additional_features = additional_features.unsqueeze(dim=0)
        else:
            # ['CLS'] token
            embedded_sentences = embedded_sentences[:, :, 0, :]
            embedded_sentences = self.dropout(embedded_sentences)
            batch_size, num_sentences, _ = embedded_sentences.size()
            sent_mask = (mask.sum(dim=2) != 0)
            embedded_sentences = self.self_attn(embedded_sentences, sent_mask)

        if additional_features is not None:
            embedded_sentences = torch.cat((embedded_sentences, additional_features), dim=-1)

        label_logits = self.time_distributed_aggregate_feedforward(embedded_sentences)
        # label_logits: batch_size, num_sentences, num_labels

        if self.labels_are_scores:
            label_probs = label_logits
        else:
            label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        # Create output dictionary for the trainer
        # Compute loss and epoch metrics
        output_dict = {"action_probs": label_probs}

        # =====================================================================

        if self.with_crf:
            # Layer 4 = CRF layer across labels of sentences in an abstract
            mask_sentences = (labels != -1)
            best_paths = self.crf.viterbi_tags(label_logits, mask_sentences)
            #
            # # Just get the tags and ignore the score.
            predicted_labels = [x for x, y in best_paths]
            # print(f"len(predicted_labels):{len(predicted_labels)}, (predicted_labels):{predicted_labels}")

            label_loss = 0.0
        if labels is not None:
            # Compute cross entropy loss
            flattened_logits = label_logits.view((batch_size * num_sentences), self.num_labels)
            flattened_gold = labels.contiguous().view(-1)

            if not self.with_crf:
                if flattened_logits.shape[0] == 1:
                    label_loss = self.loss(flattened_logits, flattened_gold)
                else:
                    label_loss = self.loss(flattened_logits.squeeze(), flattened_gold)
                if confidences is not None:
                    label_loss = label_loss * confidences.type_as(label_loss).view(-1)
                label_loss = label_loss.mean()
                flattened_probs = torch.softmax(flattened_logits, dim=-1)
            else:
                clamped_labels = torch.clamp(labels, min=0)
                log_likelihood = self.crf(label_logits, clamped_labels, mask_sentences)
                label_loss = -log_likelihood
                # compute categorical accuracy
                crf_label_probs = label_logits * 0.
                for i, instance_labels in enumerate(predicted_labels):
                    for j, label_id in enumerate(instance_labels):
                        crf_label_probs[i, j, label_id] = 1
                flattened_probs = crf_label_probs.view((batch_size * num_sentences), self.num_labels)

            if not self.labels_are_scores:
                evaluation_mask = (flattened_gold != -1)
                if flattened_probs.shape[0] == 1:
                    self.label_accuracy(flattened_probs.float().contiguous(), flattened_gold, mask=evaluation_mask)
                else:
                    self.label_accuracy(flattened_probs.float().contiguous(), flattened_gold.squeeze(-1), mask=evaluation_mask)

                # compute F1 per label
                self.label_f1_metrics_macro(flattened_probs, flattened_gold)
                self.label_f1_metrics_micro(flattened_probs, flattened_gold)
                self.label_f1_metric(flattened_probs, flattened_gold)
      
        
        if labels is not None:
            output_dict["loss"] = label_loss
            output_dict["golds"] = flattened_gold

        output_dict['action_logits'] = label_logits
        output_dict['logits'] = flattened_logits
        return output_dict

    def get_metrics(self, reset: bool = False):
        metric_dict = {}

        if not self.labels_are_scores:
            type_accuracy = self.label_accuracy.get_metric(reset)
            metric_dict['acc'] = type_accuracy
            micro_scores = self.label_f1_metrics_micro.get_metric(reset=False)
            macro_scores = self.label_f1_metrics_macro.get_metric(reset=False)
            label_scores = self.label_f1_metric.get_metric(reset=False)
            average_macro_F1 = 0.0
            average_micro_F1 = 0.0
            label_names = []
            for label_index in range(self.num_labels):
                label_name = self.vocab.get_token_from_index(namespace='labels', index=label_index)
                label_names.append(label_name)
            for i, label in enumerate(label_names):
                metric_dict[label + '_F1'] = label_scores[f"{label}"]
            metric_dict['avg-microF'] = micro_scores['f1_score']
            metric_dict['avg-macroF'] = macro_scores['f1_score']
        return metric_dict
