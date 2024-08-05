import codecs
import json
import numpy as np
from gensim.models import Word2Vec


# tokenizing function
def tokenize(sentence):
    result = sentence.replace('\n', '').replace('\r', '').split(' ')
    return(result)


def create_embeddings(file_name,
                      embeddings_path='temp_embeddings/embeddings.gensimmodel',
                      vocab_path='temp_embeddings/mapping.json',
                      **params):
    class SentenceGenerator(object):
        def __init__(self, filename):
            self.filename = filename

        def __iter__(self):
            for line in codecs.open(self.filename, 'rU', encoding='utf-8'):
                yield tokenize(line)

    sentences = SentenceGenerator(file_name)
    print(sentences)

    embedmodel = Word2Vec(sentences, **params)
    print("embedmodel: ", embedmodel)
    print("**params: ", params)
    embedmodel.save(embeddings_path)
    # weights = model.syn0
    # np.save(open(embeddings_path, 'wb'), weights)

    # http://stackoverflow.com/questions/35596031/gensim-word2vec-find-number-of-words-in-vocabulary
    vocab = dict([(k, v.index) for k, v in embedmodel.wv.vocab.items()])
    print(vocab)
    with open(vocab_path, 'w') as f:
        f.write(json.dumps(vocab))

    return vocab, embedmodel

# load embedding data
def load_vocab(vocab_path='temp_embeddings/mapping.json'):
    with open(vocab_path, 'r') as f:
        data = json.loads(f.read())
    word2idx = data
    idx2word = dict([(v, k) for k, v in data.items()])
    return word2idx, idx2word

def load_model(embedding_paths):
    w2v_model = Word2Vec.load(embedding_paths)
    return w2v_model