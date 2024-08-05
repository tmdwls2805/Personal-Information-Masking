import keras
from keras_preprocessing import sequence

def get_sublist(lst, indices):
    result = []
    for idx in indices:
        result.append(lst[idx])
    return result

def pad_data(sents, maxlen):
    sent = sequence.pad_sequences(sents, maxlen=maxlen, truncating='post', padding='post')
    return sent