import logging
from typing import Dict

import numpy as np
import torch
from torch.nn import Linear
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, TimeDistributed, Seq2SeqEncoder, Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import F1Measure, CategoricalAccuracy
from allennlp.modules.conditional_random_field import ConditionalRandomField

logger = logging.getLogger(__name__)

@Model.register("SeqClassificationModel")
class SeqClassificationModel(Model):
    """
    Question answering model where answers are sentences
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder = None,
                 with_crf: bool = False,
                 self_attn: Seq2SeqEncoder = None,
                 ) -> None:
        super(SeqClassificationModel, self).__init__(vocab)

        self.text_field_embedder = text_field_embedder
        self.vocab = vocab
        self.with_crf = with_crf
        self.encoder = encoder
        self.self_attn = self_attn
#        self.dropout = torch.nn.Dropout(p=bert_dropout)


        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        self.num_labels = self.vocab.get_vocab_size(namespace='labels')
        # define accuracy metrics
        self.label_accuracy = CategoricalAccuracy()
        self.label_f1_metrics = {}

        # define F1 metrics per label
        for label_index in range(self.num_labels):
            label_name = self.vocab.get_token_from_index(namespace='labels', index=label_index)
            self.label_f1_metrics[label_name] = F1Measure(label_index)

        self.classifier = Linear(encoder.get_output_dim(), self.num_labels)

        self.attention = Linear(self_attn.get_output_dim(), self.num_labels)

        if self.with_crf:
            self.crf = ConditionalRandomField(
                self.num_labels, constraints=None,
                include_start_end_transitions=True
            )

    def forward(self,  # type: ignore
                sentences: torch.LongTensor,
                labels: torch.IntTensor = None,
                confidences: torch.Tensor = None,
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
#        embedded_sentences = self.dropout(embedded_sentences)
        
        encoded_text = self.encoder(embedded_sentences, mask, num_wrapping_dims=1) # batch input output
        batch_size, _, _ = encoded_text.size()
        sent_mask = (mask.sum(dim=2) != 0) #?

        encoded_text_att = self.self_attn(encoded_text, sent_mask)

        label_logits = self.attention(encoded_text_att)
        # label_logits: batch_size, num_sentences, num_labels

  
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

          
            evaluation_mask = (flattened_gold != -1)
            if flattened_probs.shape[0] == 1:
                self.label_accuracy(flattened_probs.float().contiguous(), flattened_gold, mask=evaluation_mask)
            else:
                self.label_accuracy(flattened_probs.float().contiguous(), flattened_gold.squeeze(-1), mask=evaluation_mask)

            # compute F1 per label
            for label_index in range(self.num_labels):
                label_name = self.vocab.get_token_from_index(namespace='labels', index=label_index)
                metric = self.label_f1_metrics[label_name]
                metric(flattened_probs, flattened_gold, mask=evaluation_mask)
        
        if labels is not None:
            output_dict["loss"] = label_loss
        output_dict['action_logits'] = label_logits
        return output_dict

    def get_metrics(self, reset: bool = False):
        metric_dict = {}

        
        type_accuracy = self.label_accuracy.get_metric(reset)
        metric_dict['acc'] = type_accuracy

        average_F1 = 0.0
        for name, metric in self.label_f1_metrics.items():
            metric_val = metric.get_metric(reset)
            metric_dict[name + 'F'] = metric_val["f1"]
            average_F1 += metric_val["f1"]

        average_F1 /= len(self.label_f1_metrics.items())
        metric_dict['avgF'] = average_F1

        return metric_dict
