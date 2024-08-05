from transformers import BertModel
import torch.nn as nn
from torchcrf import CRF
import torch.nn.functional as F
import torch

log_soft = F.log_softmax


class BertCRF(nn.Module):
    def __init__(self, config, num_labels):
        super(BertCRF, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased") # bert multilingual model load
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, token_type_ids, attn_masks, labels=None):
        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attn_masks)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        emission = self.classifier(sequence_output)
        attn_masks = attn_masks.type(torch.uint8)
        if labels is not None:
            # log_soft를 쓰면 더 빠르게 loss를 감소시킬 수 있음
            loss = -self.crf(log_soft(emission, 2), labels, mask=attn_masks, reduction='mean')
            # loss = -self.crf(emission, labels)
            return loss
        else:
            prediction = self.crf.decode(emission)
            return prediction