from collections import Counter
import re

def read(path):
    file = open(path, 'r', encoding='utf-8-sig')
    sentences = []
    sentence_post = []
    sentence_ners = []
    this_sentence = []
    this_pos = []
    this_ner = []
    vocab = Counter()
    pos_vocab = Counter()
    ner_vocab = Counter()
    for line in file:
        if len(line) == 0 or line[0] == "\n":
            if len(this_sentence) > 0:
                sentences.append(this_sentence)
                sentence_post.append(this_pos)
                sentence_ners.append(this_ner)
                this_sentence = []
                this_pos = []
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
        # print(splits[0], splits[1], splits[2])
        this_pos.append(splits[-2])
        pos = splits[-2]
        pos_vocab[pos] = pos_vocab[pos] + 1
    print("문장 개수 : ", len(sentences))

    return sentences, sentence_post, sentence_ners, vocab, pos_vocab, ner_vocab