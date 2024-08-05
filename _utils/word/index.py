
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