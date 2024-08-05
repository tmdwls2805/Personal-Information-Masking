from transformers import BertModel
import torch.nn as nn
from torchcrf import CRF
import torch.nn.functional as F
import torch

log_soft = F.log_softmax
loss_fnt = nn.CrossEntropyLoss()


class Kobert(nn.Module):
    def __init__(self, config, num_labels):
        super(Kobert, self).__init__()
        self.num_labels = num_labels
        self.kobert = BertModel.from_pretrained('monologg/kobert')
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids, token_type_ids, attn_masks, labels=None):
        outputs = self.kobert(input_ids=input_ids,
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


class KobertCRF(nn.Module):
    def __init__(self, config, num_labels):
        super(KobertCRF, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('monologg/kobert')
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, token_type_ids, attn_masks, labels=None):
        # attention_mask = input_ids.ne(0).float()
        outputs = self.bert(input_ids=input_ids,
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


class KobertBiLSTMCRF(nn.Module):
    def __init__(self, config, num_labels):
        super(KobertBiLSTMCRF, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('monologg/kobert')
        self.dropout = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, token_type_ids, attn_masks, labels=None, using_pack_sequence=True):
        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attn_masks)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        outputs, hc = self.lstm(sequence_output)
        emission = self.classifier(outputs)
        attn_masks = attn_masks.type(torch.uint8)
        if labels is not None:
            # loss = -self.crf(log_soft(emission, 2), labels)
            loss = -self.crf(log_soft(emission, 2), labels, mask=attn_masks, reduction='mean')
            # loss = -self.crf(emission, labels)
            return loss
        else:
            # prediction = self.crf.decode(emission, mask=attn_masks)
            prediction = self.crf.decode(emission)
            return prediction


class KobertBiGRUCRF(nn.Module):
    def __init__(self, config, num_labels):
        super(KobertBiGRUCRF, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('monologg/kobert')
        self.dropout = nn.Dropout(config.dropout)
        self.gru = nn.GRU(config.hidden_size, config.hidden_size // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, token_type_ids, attn_masks, labels=None, using_pack_sequence=True):
        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attn_masks)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        outputs, hc = self.gru(sequence_output)
        emission = self.classifier(outputs)
        attn_masks = attn_masks.type(torch.uint8)
        if labels is not None:
            # loss = -self.crf(log_soft(emission, 2), labels)
            loss = -self.crf(log_soft(emission, 2), labels, mask=attn_masks, reduction='mean')
            # loss = -self.crf(emission, labels)
            return loss
        else:
            # prediction = self.crf.decode(emission, mask=attn_masks)
            prediction = self.crf.decode(emission)
            return prediction



