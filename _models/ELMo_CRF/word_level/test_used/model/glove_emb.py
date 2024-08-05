# Word-Level, glove_emb elmo 모델
# Word와 POS Tag 모두 glove embedding 수행한 모델

import os
from _utils.word import index, eval
from _utils.common import readData_POS, splitPad, save
from _utils.common.config import Config
from _embeddingModels.glove import glove
import numpy as np

config = Config(json_path="../config.json")

absolute_path = config.absolute_path
input_path = absolute_path + config.data_dir
epochs = config.epochs

# 입력 데이터 이름 (_inputData 폴더)
inputData = config.training_data
train_data_path = input_path + inputData

# 테스트 데이터 이름 (_inputData 폴더)
testData = config.test_data
test_data_path = input_path + testData

# 모델, weight, 데이터 저장 파일 경로
model_save_file_path = absolute_path + '_save/Elmo-CRF/word_level/test_used/model/glove_emb/'
data_save_file_path = absolute_path + '_save/Elmo-CRF/word_level/test_used/data/glove_emb/'
embedding_save_file_path = absolute_path +'_save/embeddings/' + inputData + '/glove_emb/'
model_save = model_save_file_path + inputData + "/" + inputData + '_' + str(epochs) + "_glove_emb.json"
model_save_weights = model_save_file_path + inputData + "/" + inputData + '_' + str(epochs) + "_glove_emb.h5"
data_save = data_save_file_path + inputData + "/" + inputData + '_' + str(epochs) + "_{0}.npy"

# 임베딩 문장 모델, text, 임베딩 vocab 저장 파일 경로
embedding_sent_text = embedding_save_file_path + 'sent_text.txt'
embedding_paths = embedding_save_file_path + 'text_embeddings.gensimmodel'
vocab_paths = embedding_save_file_path + 'text_mapping.json'

# 임베딩 형태소 태그 모델, POS_text, 임베딩 pos_vocab 저장 파일 경로
embedding_pos_text = embedding_save_file_path + 'pos_text.txt'
pos_embedding_paths = embedding_save_file_path + 'pos_embeddings.gensimmodel'
pos_vocab_paths = embedding_save_file_path + 'pos_mapping.json'


if not os.path.exists(model_save_file_path + inputData):
    print("no model_save_file_path")
    os.makedirs(model_save_file_path + inputData)
    exit()

if not os.path.exists(data_save_file_path + inputData):
    print("no data_save_file_path")
    os.makedirs(data_save_file_path + inputData)
    exit()

if not os.path.exists(embedding_save_file_path):
    print("no embedding_save_file_path")
    os.makedirs(embedding_save_file_path)
    exit()

print("input data:", inputData +".txt" '\n')
print("------------------------------------------------------------------------------", '\n')

# 학습 데이터 내 각 문장, 학습 데이터 내 각 문장에서 사용된 형태소 태그, 학습 데이터 내 각 문장에서 사용된 개체명,
# 학습 데이터 단어 집합, 학습 데이터 형태소 태그 집합, 학습 데이터 개체명 태그 집합
train_sentences, train_sentence_post, train_sentence_ners, train_vocab, train_pos_vocab, train_ner_vocab = readData_POS.read(train_data_path+".txt")

# 테스트 데이터 내 각 문장, 테스트 데이터 내 각 문장에서 사용된 형태소 태그, 테스트 데이터 내 각 문장에서 사용된 개체명,
# 테스트 데이터 단어 집합, 테스트 데이터 형태소 태그 집합, 테스트 데이터 개체명 태그 집합
test_sentences, test_sentence_post, test_sentence_ners, test_vocab, test_pos_vocab, test_ner_vocab = readData_POS.read(test_data_path+".txt")

print("train_총 단어개수: ", len(train_vocab))
print("train_sentences (대표 3개만 보기): ", train_sentences[:3])
print("train_Vocab(단어 집합)", train_vocab)
print("train_POS 태그: ", train_pos_vocab)
print("train_NER 태그: ", train_ner_vocab, '\n')


print("test_총 단어개수: ", len(test_vocab))
print("test_sentences (대표 3개만 보기): ", test_sentences[:3])
# vocab 안의 단어들을 빈도수 순으로 정렬
print("test_Vocab(단어 집합)", test_vocab)
print("test_POS 태그: ", test_pos_vocab)
print("test_NER 태그: ", test_ner_vocab, '\n')

print("------------------------------------------------------------------------------", '\n')


# index
word_to_index, index_to_word = index.make_index(train_vocab, 'word')
pos_to_index, index_to_pos = index.make_index(train_pos_vocab, 'pos')
ner_to_index, index_to_ner = index.make_index(train_ner_vocab, 'ner')

train_sentence_text_idx = index.index_sents(train_sentences, word_to_index)
train_sentence_post_idx = index.index_sents(train_sentence_post, pos_to_index)
train_sentence_ners_idx = index.index_sents(train_sentence_ners, ner_to_index)
print("train_sentence_text_idx: ", train_sentence_text_idx[:2])
print("train_sentence_post_idx: ", train_sentence_post_idx[:2])
print("train_sentence_ners_idx: ", train_sentence_ners_idx[:2], '\n')

test_sentence_text_idx = index.index_sents(test_sentences, word_to_index)
test_sentence_post_idx = index.index_sents(test_sentence_post, pos_to_index)
test_sentence_ners_idx = index.index_sents(test_sentence_ners, ner_to_index)
print("test_sentence_text_idx: ", test_sentence_text_idx[:2])
print("test_sentence_post_idx: ", test_sentence_post_idx[:2])
print("test_sentence_ners_idx: ", test_sentence_ners_idx[:2], '\n')

print("------------------------------------------------------------------------------", '\n')

# Split train & test sets, Pad data
train_idx = [i for i in range(len(train_sentences))]
test_idx = [i for i in range(len(test_sentences))]

X_train_sents = splitPad.get_sublist(train_sentence_text_idx, train_idx)
X_test_sents = splitPad.get_sublist(test_sentence_text_idx, test_idx)
X_train_pos = splitPad.get_sublist(train_sentence_post_idx, train_idx)
X_test_pos = splitPad.get_sublist(test_sentence_post_idx, test_idx)
y_train_ner = splitPad.get_sublist(train_sentence_ners_idx, train_idx)
y_test_ner = splitPad.get_sublist(test_sentence_ners_idx, test_idx)

max_len = config.maxlen
X_train_sents = splitPad.pad_data(X_train_sents, max_len)
X_test_sents = splitPad.pad_data(X_test_sents, max_len)
X_train_pos = splitPad.pad_data(X_train_pos, max_len)
X_test_pos = splitPad.pad_data(X_test_pos, max_len)
y_train_ner = splitPad.pad_data(y_train_ner, max_len)
y_test_ner = splitPad.pad_data(y_test_ner, max_len)

print("------------------------------------------------------------------------------", '\n')

# load embedding data
if not os.path.exists(vocab_paths):
    print("no embedding vocab file - Please make embedding vectors from _genEmbeddingVectors")
    exit()
elif not os.path.exists(embedding_paths):
    print("no embedding file - Please make embedding vectors from _genEmbeddingVectors")
    exit()
elif not os.path.exists(pos_vocab_paths):
    print("no embedding pos_vocab file - Please make embedding vectors from _genEmbeddingVectors")
    exit()
elif not os.path.exists(pos_embedding_paths):
    print("no pos embedding file - Please make embedding vectors from _genEmbeddingVectors")
    exit()
else:
    glove_vocab = glove.load_vocab(vocab_paths)
    glove_model = glove.load_model(embedding_paths)
    glove_pvocab = glove.load_vocab(pos_vocab_paths)
    glove_pmodel = glove.load_model(pos_embedding_paths)


print("------------------------------------------------------------------------------", '\n')
# embedding matrices
print("creating embedding matrices...\n")
MAX_VOCAB = len(list(word_to_index.keys()))
TAG_VOCAB = len(list(index_to_pos.keys()))
EMBEDDING_SIZE = 100 # from default gensim model, see preprocessing.ipynb
word_embedding_matrix = np.zeros((MAX_VOCAB, EMBEDDING_SIZE))

for word in word_to_index.keys():
    # get the word vector from the embedding model
    # if it's there (check against vocab list)
    if word in glove_vocab.values():
        # print("word:", word)
        for key, value in glove_vocab.items():
            if value == word:
                temp = key
        # print("key:",  temp)
        # get the word vector
        word_vector = glove_model[temp]
        # slot it in at the proper index
        # print(word, word_vector)
    # print("word_vector:", word, word_vector)
    word_embedding_matrix[word_to_index[word]] = word_vector

print("word_embedding_matrix:", len(word_embedding_matrix))
print("word_embedding_matrix:", word_embedding_matrix[2])


pos_embedding_matrix = np.zeros((TAG_VOCAB, EMBEDDING_SIZE))

for pos in pos_to_index.keys():
    # get the word vector from the embedding model
    # if it's there (check against vocab list)
    if pos in glove_pvocab.values():
        # get the word vector
        for key, value in glove_pvocab.items():
            if value == pos:
                temp = key
        # print("key:",  temp)
        # get the word vector
        pos_vector = glove_pmodel[temp]
        # slot it in at the proper index
        # print(word, word_vector)
    # print("word_vector:", word, word_vector)
    pos_embedding_matrix[pos_to_index[pos]] = pos_vector


print("------------------------------------------------------------------------------", '\n')

MAX_VOCAB = len(list(word_to_index.keys()))
TAG_VOCAB = len(list(index_to_pos.keys()))
NER_VOCAB = len(list(index_to_ner.keys()))
print("MAX_VOCAB:", MAX_VOCAB)
print("TAG_VOCAB:", TAG_VOCAB)
print("NER_VOCAB:", NER_VOCAB)


DROPOUTRATE = 0.2
cell_clip = 5
proj_clip = 5
hidden_units_size = 200
word_dropout_rate = 0.05

# model
print('Building model...\n')
from keras.models import Model, Input
from keras.layers import concatenate, LSTM, Embedding, Dense, TimeDistributed, Bidirectional, Dropout, SpatialDropout1D, add, Lambda
from keras.constraints import MinMaxNorm
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras import backend as K

def reverse(inputs, axes=1):
    return K.reverse(inputs, axes=axes)

# text layers
txt_input = Input(shape=(max_len,), name='txt_input')
txt_embed = Embedding(MAX_VOCAB, EMBEDDING_SIZE, input_length=max_len,
                      weights=[word_embedding_matrix],
                      name='txt_embedding', trainable=True, mask_zero=False)(txt_input)
txt_drpot = Dropout(DROPOUTRATE, name='txt_dropout')(txt_embed)
#txt_lstml = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1), name='txt_bidirectional')(txt_drpot)

# pos layers
pos_input = Input(shape=(max_len,), name='pos_input')
pos_embed = Embedding(TAG_VOCAB, EMBEDDING_SIZE, input_length=max_len,
                      weights=[pos_embedding_matrix],
                      name='pos_embedding', trainable=True)(pos_input)
pos_drpot = Dropout(DROPOUTRATE, name='pos_dropout')(pos_embed)
#pos_lstml = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1), name='pos_bidirectional')(pos_drpot)

# Concatenate input
model = concatenate([txt_drpot, pos_drpot], axis=2)
#model = concatenate([txt_lstml, pos_lstml], axis=2)

lstm_inputs = SpatialDropout1D(0.3)(model)

# Reversed input for backward LSTMs
re_lstm_inputs = Lambda(function=reverse)(lstm_inputs)
mask = Lambda(function=reverse)(lstm_inputs)


# Deep Layers

# Forward LSTMs
lstm = LSTM(units=100, return_sequences=True, activation="tanh", recurrent_activation='sigmoid',
            kernel_constraint=MinMaxNorm(-1 * cell_clip, cell_clip),
            recurrent_constraint=MinMaxNorm(-1 * cell_clip, cell_clip))(lstm_inputs)

# Projection to hidden_units_size
proj = TimeDistributed(Dense(hidden_units_size, activation='linear',kernel_constraint=MinMaxNorm(-1 * proj_clip, proj_clip)))(lstm)

# Merge Bi-LSTMs feature vectors with the previous ones
lstm_inputs = add([proj, lstm_inputs], name='f_block_{}'.format(1))

# Apply variational drop-out between BI-LSTM layers
outputs = SpatialDropout1D(DROPOUTRATE)(lstm_inputs)

# Backward LSTMs
re_lstm = LSTM(units=100, return_sequences=True, activation='tanh',recurrent_activation='sigmoid',
               kernel_constraint=MinMaxNorm(-1 * cell_clip, cell_clip),
               recurrent_constraint=MinMaxNorm(-1 * cell_clip, cell_clip))(re_lstm_inputs)

# Projection to hidden_units_size
re_proj = TimeDistributed(Dense(hidden_units_size, activation='linear', kernel_constraint=MinMaxNorm(-1 * proj_clip, proj_clip)))(re_lstm)
# Merge Bi-LSTMs feature vectors with the previous ones

re_lstm_inputs = add([re_proj, re_lstm_inputs], name='b_block_{}'.format(1))

# Apply variational drop-out between BI-LSTM layers
re_lstm_inputs = SpatialDropout1D(DROPOUTRATE)(re_lstm_inputs)

# Reverse backward LSTMs' outputs = Make it forward again
re_outputs = Lambda(function=reverse, name="reverse")(re_lstm_inputs)

output = concatenate([outputs, re_outputs])

# output layer
out = TimeDistributed(Dense(50, activation='relu'))(output)  # softmax output layer
crf = CRF(NER_VOCAB)
mrg_chain = crf(out)


model = Model(inputs=[txt_input, pos_input], outputs=mrg_chain)
model.compile(optimizer="adam", loss=crf_loss, metrics=[crf_viterbi_accuracy])
                         #adam, rmsprop




from keras.utils import np_utils
y_train2 = np_utils.to_categorical(y_train_ner)

epochs = config.epochs
data_name = config.data_name

model.summary()
history = model.fit([X_train_sents, X_train_pos], y_train2, batch_size=config.batch_size, epochs=epochs, validation_split=config.validation_rate, verbose=1)

y_test2 = np_utils.to_categorical(y_test_ner)
print("\n 테스트 정확도: %.4f" % (model.evaluate([X_test_sents, X_test_pos], y_test2)[1]), '\n')


print("------------------------------------------------------------------------------", '\n')
# save

print('save model & parameters.................')
save.modelsave(model, model_save, model_save_weights)

saves = [X_test_sents, X_test_pos, y_test_ner, word_to_index, ner_to_index, pos_to_index, index_to_word, index_to_ner]
names = ['X_test_sents', 'X_test_pos', 'y_test_ner', 'word_to_index', 'ner_to_index', 'pos_to_index', 'index_to_word', 'index_to_ner']

save.numpy_save(saves, names, data_save)


# evaluation

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

tmp_accuracy = model.evaluate([X_test_sents, X_test_pos], y_test2)[1]

report, accuracy = classification_report(test_tags, pred_tags, tmp_accuracy)
print("\n", report)
classification_report_to_txt(config, accuracy, "glove", "../result/", report)