from transformers import ElectraModel
import torch.nn as nn
from torchcrf import CRF
import torch.nn.functional as F
import torch

log_soft = F.log_softmax
loss_fnt = nn.CrossEntropyLoss()  # ignore_index 지정?


class KoElectra(nn.Module):
    def __init__(self, config, num_labels):
        super(KoElectra, self).__init__()
        self.num_labels = num_labels
        self.electra = ElectraModel.from_pretrained("monologg/koelectra-base-v2-discriminator") # base-v2도 test해봐야함
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids, token_type_ids, attn_masks, labels=None):
        outputs = self.electra(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attn_masks)

        last_encoder_layer = outputs[0]
        last_encoder_layer = self.dropout(last_encoder_layer)
        logits = self.classifier(last_encoder_layer)

        if labels is not None:
            logits = logits.view(-1, self.num_labels)
            labels = labels.view(-1).long()
            loss = loss_fnt(logits, labels)
            return loss
        else:
            return logits


class KoElectraCRF(nn.Module):
    def __init__(self, config, num_labels):
        super(KoElectraCRF, self).__init__()
        self.num_labels = num_labels
        self.electra = ElectraModel.from_pretrained("monologg/koelectra-base-v2-discriminator")
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, token_type_ids, attn_masks, labels=None):
        outputs = self.electra(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attn_masks)

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