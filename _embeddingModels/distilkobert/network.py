from transformers import DistilBertModel
import torch.nn as nn
from torchcrf import CRF
import torch.nn.functional as F
import torch


log_soft = F.log_softmax


class DistilKobertCRF(nn.Module):
    def __init__(self, config, num_labels):
        super(DistilKobertCRF, self).__init__()
        self.num_labels = num_labels
        self.bert = DistilBertModel.from_pretrained('monologg/distilkobert')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, attn_masks, labels=None):
        # attention_mask = input_ids.ne(0).float()
        outputs = self.bert(input_ids, attn_masks)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        emission = self.classifier(sequence_output)
        attn_masks = attn_masks.type(torch.uint8)
        if labels is not None:
            loss = -self.crf(log_soft(emission, 2), labels, mask=attn_masks, reduction='mean')
            # loss = -self.crf(emission, labels)
            return loss
        else:
            # prediction = self.crf.decode(emission, mask=attn_masks)
            prediction = self.crf.decode(emission)
            return prediction