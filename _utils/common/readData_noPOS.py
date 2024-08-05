from collections import Counter
import re

def read(path):
    file = open(path, 'r', encoding='utf-8')
    sentences = []
    sentence_ners = []
    this_sentence = []
    this_ner = []
    vocab = Counter()
    ner_vocab = Counter()
    for line in file:
        if len(line) == 0 or line[0] == "\n":
            if len(this_sentence) > 0:
                sentences.append(this_sentence)
                sentence_ners.append(this_ner)
                this_sentence = []
                this_ner = []
            continue
        splits = line.split(' ')
        splits[-1] = re.sub(r'\n', '', splits[-1])
        word = splits[0]
        vocab[word] = vocab[word] + 1
        this_sentence.append(word)
        this_ner.append(splits[-1])
        ner = splits[-1]
        ner_vocab[ner] = ner_vocab[ner] + 1


    return sentences, sentence_ners, vocab, ner_vocab