import numpy as np
from .split_char import split_char_convert

def make_index(vocab, tags):
    word_sorted = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    if tags == 'word' or tags == 'pos':
        word_to_index = {w: i + 2 for i, (w, n) in enumerate(word_sorted)}
        word_to_index['PAD'] = 0  # 패딩을 위해 인덱스 0 할당
        word_to_index['OOV'] = 1  # 모르는 단어을 위해 인덱스 1 할당
        index_to_word = {i: w for w, i in word_to_index.items()}
    else:
        word_to_index = {w: i for i, (w, n) in enumerate(word_sorted)}
        index_to_word = {i: w for w, i in word_to_index.items()}
    return word_to_index, index_to_word

def make_char_index(vocab, tags):
    chars = set([w_i for w in vocab for w_i in w])
    char_to_index = {c: i + 2 for i, c in enumerate(chars)}
    char_to_index["OOV"] = 1
    char_to_index["PAD"] = 0
    index_to_char = {i: w for w, i in char_to_index.items()}
    return char_to_index, index_to_char


def make_jamo_index(vocab, tags):
    chars = set([w_i for w in vocab for w_i in w])
    vocab_character = set()
    for ch in chars:
        vocab_character.update(split_char_convert(ch))

    character = {c: i + 2 for i, c in enumerate(vocab_character)}
    character_to_index = {c: i + 2 for i, c in enumerate(character)}
    character_to_index["UNK"] = 1
    character_to_index["PAD"] = 0
    index_to_character = {i: w for w, i in character_to_index.items()}
    return character_to_index, index_to_character


def index_sents(sents, vocab, testing=0):
    if testing==1:
        print("starting vectorize_sents()...")
    vectors = []
    # iterate thru sents
    for sent in sents:
        sent_vect = []
        for word in sent:
            if word in vocab.keys():
                idx = vocab[word]
                sent_vect.append(idx)
            else: # out of max_vocab range or OOV
                sent_vect.append(vocab['OOV'])
        vectors.append(sent_vect)
    return(vectors)

def index_chars(sents, char, max_len, max_len_char):
    vectors = []
    for sent in sents:
        sent_seq = []
        for i in range(max_len):
            word_seq = []
            for j in range(max_len_char):
                try:
                    # print("sents:", sentence[i][j])
                    # print("char_to_index:",char_to_index.get(sentence[i][j]))
                    word_seq.append(char.get(sent[i][j]))
                except:
                    # print("sents: O")
                    # print("char_to_index: PAD")
                    word_seq.append(char.get('PAD'))
            sent_seq.append(word_seq)
        vectors.append(np.array(sent_seq))
    return (vectors)

def index_jamos(sents, char, max_len, max_len_char):
    vectors = []
    for sent in sents:
        sent_seq = []
        for i in range(max_len):
            word_seq = []
            for j in range(max_len_char):
                try:
                    for jamo in split_char_convert(sent[i][j]):
                        word_seq.append(char.get(jamo))
                    if len(split_char_convert(sent[i][j])) == 1:
                        word_seq.append(char.get('PAD'))
                        word_seq.append(char.get('PAD'))
                    if len(split_char_convert(sent[i][j])) == 2:
                        word_seq.append(char.get('PAD'))
                except:
                    word_seq.append(char.get('PAD'))
                    word_seq.append(char.get('PAD'))
                    word_seq.append(char.get('PAD'))
            sent_seq.append(word_seq)
        vectors.append(np.array(sent_seq))
    return (vectors)