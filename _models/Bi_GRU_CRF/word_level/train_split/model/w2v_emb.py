import os
from _utils.word import index, eval
from _utils.common import readData_POS, splitPad, save
from _utils.common.config import Config
from sklearn.model_selection import train_test_split
from _embeddingModels.gensim import word2vec
import numpy as np

config = Config(json_path="../config.json")

absolute_path = config.absolute_path
input_path = absolute_path + config.data_dir

inputData = config.training_data
train_data_path = input_path + inputData

epochs = config.epochs

model_save_path = absolute_path + '_save/Bi-GRU-CRF/word_level/train_split/model/w2v_emb/'
data_save_path = absolute_path + '_save/Bi-GRU-CRF/word_level/train_split/data/w2v_emb/'
embedding_save_path = absolute_path + '_save/embeddings/' + inputData + '/w2v_emb/'
model_save = model_save_path + inputData + '/' + inputData + '_' + str(epochs) + '_w2v_emb.json'
model_save_weights = model_save_path + inputData + '/' + inputData + '_' + str(epochs) + '_w2v_emb.h5'
data_save = data_save_path + inputData + '/' + inputData + '_' + str(epochs) + '_{0}.npy'

embedding_sent_text = embedding_save_path + 'sent_text.txt'
embedding_paths = embedding_save_path + 'text_embeddings.gensimmodel'
vocab_paths = embedding_save_path + 'text_mapping.json'

embedding_pos_text = embedding_save_path + 'pos_text.txt'
pos_embedding_paths = embedding_save_path + 'pos_embeddings.gensimmodel'
pos_vocab_paths = embedding_save_path + 'pos_mapping.json'

if not os.path.exists(model_save_path + inputData):
    print("model_save_file_path is not exist")
    os.makedirs(model_save_path + inputData)
    exit()

if not os.path.exists(data_save_path + inputData):
    print("data_save_file_path is not exist")
    os.makedirs(data_save_path + inputData)
    exit()

if not os.path.exists(embedding_save_path):
    print("embedding_save_file_path is not exist")
    exit()


sentences, sentence_post, sentence_ners, vocab, pos_vocab, ner_vocab = readData_POS.read(train_data_path+".txt")

print("총 단어개수: ", len(vocab))
print("sentences (대표 3개만 보기): ", sentences[:3])
print("Vocab(단어 집합)", vocab)
print("POS 태그: ", pos_vocab)
print("NER 태그: ", ner_vocab, '\n')

# Index
word_to_index, index_to_word = index.make_index(vocab, 'word')
pos_to_index, index_to_pos = index.make_index(pos_vocab, 'pos')
ner_to_index, index_to_ner = index.make_index(ner_vocab, 'ner')

sentence_text_idx = index.index_sents(sentences, word_to_index)
sentence_post_idx = index.index_sents(sentence_post, pos_to_index)
sentence_ners_idx = index.index_sents(sentence_ners, ner_to_index)

indices = [i for i in range(len(sentences))]
train_idx, test_idx, X_train_pos, X_test_pos = train_test_split(indices, sentence_post_idx, test_size=config.test_rate, random_state=1)

X_train_sents = splitPad.get_sublist(sentence_text_idx, train_idx)
X_test_sents = splitPad.get_sublist(sentence_text_idx, test_idx)
y_train_ner = splitPad.get_sublist(sentence_ners_idx, train_idx)
y_test_ner = splitPad.get_sublist(sentence_ners_idx, test_idx)

max_len = max([len(i) for i in X_train_sents])
avg_len = (sum(map(len, X_train_sents))/len(X_train_sents))
print('문장 최대 길이 : ', max_len)
print('문장 평균 길이 : ', int(avg_len))

max_len = config.maxlen

X_train_sents = splitPad.pad_data(X_train_sents, max_len)
X_test_sents = splitPad.pad_data(X_test_sents, max_len)
X_train_pos = splitPad.pad_data(X_train_pos, max_len)
X_test_pos = splitPad.pad_data(X_test_pos, max_len)
y_train_ner = splitPad.pad_data(y_train_ner, max_len)
y_test_ner = splitPad.pad_data(y_test_ner, max_len)

# Load embedding data
if not os.path.exists(vocab_paths):
    print("no embedding vocab file - Please make embedding vectors from _genEmbeddingVectors")
    exit()
elif not os.path.exists(embedding_paths):
    print("no embedding file - Please make embedding vectors from _genEmbeddingVectors")
    exit()
else:
    w2v_vocab, _ = word2vec.load_vocab(vocab_paths)
    w2v_model = word2vec.load_model(embedding_paths)
    w2v_pvocab, _ = word2vec.load_vocab(pos_vocab_paths)
    w2v_pmodel = word2vec.load_model(pos_embedding_paths)

# Embedding matrices
print("creating embedding matrices...\n")
MAX_VOCAB = len(list(word_to_index.keys()))
POS_VOCAB = len(list(pos_to_index.keys()))
NER_VOCAB = len(list(index_to_ner.keys()))
EMBEDDING_SIZE = 100 # from default gensim model
word_embedding_matrix = np.zeros((MAX_VOCAB, EMBEDDING_SIZE))
pos_embedding_matrix = np.zeros((POS_VOCAB, EMBEDDING_SIZE))

for word in word_to_index.keys():
    # get the word vector from the embedding model
    # if it's there (check against vocab list)
    if word in w2v_vocab:
        # get the word vector
        word_vector = w2v_model[word]
        # slot it in at the proper index
    word_embedding_matrix[word_to_index[word]] = word_vector

for pos in pos_to_index.keys():
    # get the word vector from the embedding model
    # if it's there (check against vocab list)
    if pos in w2v_pvocab:
        # get the word vector
        pos_vector = w2v_pmodel[pos]
        # slot it in at the proper index
    pos_embedding_matrix[pos_to_index[pos]] = pos_vector

DROPOUTRATE = 0.2

# Model
print('Building model...\n')
from keras.models import Model, Input
from keras.layers import concatenate, GRU, LSTM, Embedding, Dense, TimeDistributed, Bidirectional, Dropout, SpatialDropout1D
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

# Text layers
txt_input = Input(shape=(max_len,), name='txt_input')
txt_embed = Embedding(MAX_VOCAB, output_dim=EMBEDDING_SIZE, input_length=max_len,
                      weights=[word_embedding_matrix],
                      name='txt_embedding', trainable=True)(txt_input)
txt_drpot = Dropout(DROPOUTRATE, name='txt_dropout')(txt_embed)

# POS layers
pos_input = Input(shape=(max_len,), name='pos_input')
pos_embed = Embedding(POS_VOCAB, output_dim=EMBEDDING_SIZE, input_length=max_len,
                      weights=[pos_embedding_matrix],
                      name='pos_embedding', trainable=True)(pos_input)
pos_drpot = Dropout(DROPOUTRATE, name='pos_dropout')(pos_embed)

# Concatenate input
model = concatenate([txt_drpot, pos_drpot], axis=2)
model = SpatialDropout1D(DROPOUTRATE)(model)

# Deep layers
model = Bidirectional(GRU(units=128, return_sequences=True, recurrent_dropout=0.1))(model)

# Output layers
out = TimeDistributed(Dense(128, activation='relu'))(model)
crf = CRF(NER_VOCAB)
mrg_chain = crf(out)

model = Model(inputs=[txt_input, pos_input], outputs=mrg_chain)
model.compile(optimizer='rmsprop', loss=crf_loss, metrics=[crf_viterbi_accuracy])
                        #adam



from keras.utils import np_utils
y_train2 = np_utils.to_categorical(y_train_ner)


model.summary()
history = model.fit([X_train_sents, X_train_pos], y_train2, batch_size=config.batch_size, epochs=epochs, validation_split=config.validation_rate, verbose=1)

y_test2 = np_utils.to_categorical(y_test_ner)
tmp_accuracy = model.evaluate([X_test_sents, X_test_pos], y_test2)[1]

print("\n 테스트 정확도: %.4f" % (model.evaluate([X_test_sents, X_test_pos], y_test2)[1]), '\n')


print("------------------------------------------------------------------------------", '\n')
# save

print('save model & parameters.................')
save.modelsave(model, model_save, model_save_weights)

saves = [X_test_sents, X_test_pos, y_test_ner, word_to_index, ner_to_index, pos_to_index, index_to_word, index_to_ner]
names = ['X_test_sents', 'X_test_pos', 'y_test_ner', 'word_to_index', 'ner_to_index', 'pos_to_index', 'index_to_word', 'index_to_ner']

save.numpy_save(saves, names, data_save)


print('evaluation.................', '\n')
i=2 # 확인하고 싶은 테스트용 샘플의 인덱스.
this_txt = splitPad.pad_data([X_test_sents[i]], max_len)
this_pos = splitPad.pad_data([X_test_pos[i]], max_len)

eval.testData_POS(i, model, this_txt, this_pos, y_test2, index_to_word, index_to_ner, X_test_sents)

y_predicted = model.predict([X_test_sents, X_test_pos])
pred_tags = eval.sequences_to_tag(y_predicted, index_to_ner)
test_tags = eval.sequences_to_tag(y_test2, index_to_ner)

from _utils.common.classification_report import classification_report
from _utils.gru.utils import classification_report_to_txt

report, accuracy = classification_report(test_tags, pred_tags, tmp_accuracy)
print("\n", report)
classification_report_to_txt(config, accuracy, "w2v", "../result/", report)
