import re
import torch
import pickle
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np
import logging
from torch.utils import data
logger = logging.getLogger(__name__)


class NER_Dataset(data.Dataset):
    def __init__(self, config, tag2idx, sentences, labels, tokenizer):
        self.tag2idx = tag2idx
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        bert_tokens = []
        orig_to_tok_map = []
        bert_tokens.append('[CLS]')
        for i, token in enumerate(sentence):
            orig_to_tok_map.append(len(bert_tokens))
            new_token = self.tokenizer.tokenize(token)
            bert_tokens.extend(new_token)
        bert_tokens.append('[SEP]')
        token_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)

        return token_ids, len(token_ids), orig_to_tok_map, self.sentences[idx]

    def make_training_inputs(self):
        sequence_segment_id = 0
        tokenized_sents = []
        tokenized_sents_label = []
        token_type_ids = []
        maxlen = self.config.maxlen
        for sentence, labels in zip(self.sentences, self.labels):
            token_list = ["[CLS]"]
            label_list = ["O"]
            for word, label in zip(sentence, labels):
                token = self.tokenizer.tokenize(word)
                token_list.extend(token)
                for i, _ in enumerate(token):
                    if i == 0:
                        label_list.append(label)
                    else:
                        label_list.append(label)
            token_list.append("[SEP]")
            label_list.append("O")
            token_type_ids.append([sequence_segment_id] * len(token_list))
            tokenized_sents.append(token_list)
            tokenized_sents_label.append(label_list)

        pad_token_id = self.tokenizer.pad_token_id # 0:[PAD]

        token_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(tok) for tok in tokenized_sents],
                                  maxlen=maxlen, dtype="long", truncating="post", padding="post", value=pad_token_id)
        token_labels = pad_sequences([[self.tag2idx.get(I) for I in label] for label in tokenized_sents_label],
                                     maxlen=maxlen, padding="post", value=self.tag2idx["O"], dtype="long", truncating="post")
        token_type_ids = pad_sequences([token_type for token_type in token_type_ids], maxlen=maxlen, dtype="long",
                                       truncating="post", padding="post", value=0)

        # tokenizer의 pad_token_id인 0을 제외하여 input mask 생성
        masks = [[float(i > 0) for i in ids] for ids in token_ids]

        return token_ids, token_type_ids, masks, token_labels

    def make_test_inputs(self):
        sequence_segment_id = 0
        tokenized_sents = []
        tokenized_sents_label = []
        token_type_ids = []
        maxlen = self.config.maxlen
        for sentence, labels in zip(self.sentences, self.labels):
            token_list = ["[CLS]"]
            label_list = ["O"]
            for word, label in zip(sentence, labels):
                token = self.tokenizer.tokenize(word)
                token_list.extend(token)
                for i, _ in enumerate(token):
                    if i == 0:
                        label_list.append(label)
                    else:
                        label_list.append(label)
            token_list.append("[SEP]")
            label_list.append("O")
            token_type_ids.append([sequence_segment_id] * len(token_list))
            tokenized_sents.append(token_list)
            tokenized_sents_label.append(label_list)

        pad_token_id = self.tokenizer.pad_token_id  # 0:[PAD]

        token_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(tok) for tok in tokenized_sents],
                                  maxlen=maxlen, dtype="long", truncating="post", padding="post", value=pad_token_id)
        token_labels = pad_sequences([[self.tag2idx.get(I) for I in label] for label in tokenized_sents_label],
                                     maxlen=maxlen, padding="post", value=self.tag2idx["O"], dtype="long",
                                     truncating="post")
        token_type_ids = pad_sequences([token_type for token_type in token_type_ids], maxlen=maxlen, dtype="long",
                                       truncating="post", padding="post", value=0)
        # tokenizer의 pad_token_id인 1을 제외하여 input mask 생성
        masks = [[float(i > 0) for i in ids] for ids in token_ids]

        return token_ids, token_type_ids, masks, token_labels


def corpus_reader(file_path, word_idx=0, label_idx=-1):
    sentences, labels = [], []
    tmp_tok, tmp_lab = [], []
    token_vocab = Counter()
    label_vocab = Counter()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if len(line) == 0 or line[0] == "\n":
                if len(tmp_tok) > 0:
                    sentences.append(tmp_tok)
                    labels.append(tmp_lab)
                    tmp_tok, tmp_lab = [], []
                continue
            splits = line.split(' ')
            splits[label_idx] = re.sub(r'\n', '', splits[label_idx])
            token = splits[word_idx]
            token_vocab[token] = token_vocab[token] + 1
            tmp_tok.append(token)
            tmp_lab.append(splits[label_idx])
            label = splits[label_idx]
            label_vocab[label] = label_vocab[label] + 1

    return sentences, labels, token_vocab, label_vocab

# input data들을 tensor 형태로 바꿔줌
def transform_to_tensor_dataset(config, tokens, token_type_ids, labels, masks):
    input = torch.LongTensor(tokens)
    token_type = torch.LongTensor(token_type_ids)
    mask = torch.LongTensor(masks)

    # validation을 위한 tensor data 생성
    if labels is not None:
        label = torch.LongTensor(labels)
        data = TensorDataset(input, token_type, mask, label)
    # inference를 위한 tensor data 생성
    else:
        data = TensorDataset(input, token_type, mask)

    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=config.batch_size)

    return dataloader


def generate_training_data(config, tokenizer, mode=None):
    training_data = config.absolute_path+config.data_dir+config.training_data

    # input data로부터 문장, 정답, vocab 분리
    sentences, labels, token_vocab, label_vocab = corpus_reader(training_data)
    print('TRAIN 문장 개수 : ', len(sentences))
    # 사용된 태그들 list로 만들어줌
    label_list = sorted(list(set([la for label in labels for la in label])))
    tag2idx = {t: i for i, t in enumerate(label_list)}

    # Prediction 위해 tag2idx vocab 저장
    with open('./ner_vocab_{}.pkl'.format(config.data_name), 'wb') as f:
        pickle.dump(tag2idx, f, pickle.HIGHEST_PROTOCOL)

    ner_ds = NER_Dataset(config=config, tag2idx=tag2idx, sentences=sentences, labels=labels, tokenizer=tokenizer)
    # model input data(token_ids, mask, label) 생성
    token_ids, token_type_ids, masks, labels = ner_ds.make_training_inputs()

    # test set 사용하지 않을 경우 train set split
    if mode is None:
        tr_inputs, val_inputs, tr_labels, val_labels = train_test_split(token_ids, labels, random_state=1234, test_size=0.3)
        tr_token_ids, val_token_ids, _, _ = train_test_split(token_type_ids, token_ids, random_state=1234, test_size=0.3)
        tr_masks, val_masks, _, _ = train_test_split(masks, token_ids, random_state=1234, test_size=0.3)
        tr_dataloader = transform_to_tensor_dataset(config, tr_inputs, tr_token_ids, tr_labels, tr_masks)
        val_dataloader = transform_to_tensor_dataset(config, val_inputs, val_token_ids, val_labels, val_masks)
    # test set 사용할 경우
    elif mode == "test":
        tr_inputs = token_ids
        tr_labels = labels
        tr_masks = masks
        tr_token_type_ids = token_type_ids
        val_inputs, val_token_type_ids, val_labels, val_masks = generate_test_data(config, tokenizer, tag2idx)
        tr_dataloader = transform_to_tensor_dataset(config, tr_inputs, tr_token_type_ids, tr_labels, tr_masks)
        val_dataloader = transform_to_tensor_dataset(config, val_inputs, val_token_type_ids, val_labels, val_masks)
    else:
        logger.info(" ****** Please assign mode as None(train_split) or test  ******")
        exit(1)

    return tr_dataloader, val_dataloader, label_list, tag2idx


def read_input_file(config):
    lines = []
    with open(config.test_data, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            words = line.split()
            lines.append(words)

    return lines


def generate_test_data(config, tokenizer, tag2idx):
    test_data = config.absolute_path+config.data_dir+config.test_data

    sentences, labels, _, _ = corpus_reader(test_data)
    print('TEST 문장 개수 : ', len(sentences))
    ner_ds = NER_Dataset(config=config, tag2idx=tag2idx, sentences=sentences, labels=labels, tokenizer=tokenizer)
    val_inputs, val_token_type_ids, val_masks, val_labels = ner_ds.make_test_inputs()

    return val_inputs, val_token_type_ids, val_labels, val_masks

# prediction시 bert모델 input 형태로 만들어주는 함수
def raw_processing(sents, tokenizer):
    sentences = sents
    batch = []
    offset = 0

    for s_idx, sentence in enumerate(sentences):
        bert_tokens = []
        orig_to_tok_map = []
        bert_tokens.append('[CLS]')
        begins =[]
        ends = []
        splits = sentence.split(' ')
        for tok in splits:
            token = tok
            offset = sentence.find(token, offset)
            ends.append(offset + len(token))
            offset += len(token)
            orig_to_tok_map.append(len(bert_tokens))
            new_token = tokenizer.tokenize(token)
            bert_tokens.extend(new_token)
        bert_tokens.append('[SEP]')
        token_id = tokenizer.convert_tokens_to_ids(bert_tokens)
        original_tokens = splits
        sample = (token_id, len(token_id), orig_to_tok_map, original_tokens)
        batch.append(sample)
    pad_data = pad(batch)

    return pad_data

# 예측시 input 텍스트 padding 하기 위한 함수이며 예측결과 디코딩을 위해 original token들을 리턴해줌
def pad(batch):
    get_element = lambda x: [sample[x] for sample in batch]
    seq_len = get_element(1)
    maxlen = np.array(seq_len).max()
    do_pad = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # 0: [PAD] (KoElectraTokenizer)
    tok_ids = do_pad(0, maxlen)

    attn_mask = [[float(i > 0) for i in ids] for ids in tok_ids]
    token_type_ids = [[0 for _ in range(maxlen)] for _ in range(len(tok_ids))]

    token_ids = get_element(0)
    token_ids_len = torch.LongTensor(list(map(len, token_ids)))
    _, sorted_idx = token_ids_len.sort(0, descending=True)

    tok_ids = torch.LongTensor(tok_ids)[sorted_idx]
    token_type_ids = torch.LongTensor(token_type_ids)[sorted_idx]
    attn_mask = torch.LongTensor(attn_mask)[sorted_idx]
    org_tok_map = get_element(2)
    original_tokens = get_element(-1)

    return tok_ids, token_type_ids, attn_mask, org_tok_map, list(sorted_idx.cpu().numpy()), original_tokens