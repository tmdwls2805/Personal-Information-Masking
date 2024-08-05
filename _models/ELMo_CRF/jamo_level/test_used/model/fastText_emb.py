# Char-Level, Bi-LSTM, fastText_emb POS 사용 모델
# Word, POS Tag, char 모두 fastText embedding 수행한 모델

import os
from _utils.jamo import index, eval
from _utils.common import readData_POS, splitPad, save
from _utils.common.config import Config
from _embeddingModels.gensim import fastText
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
model_save_file_path = absolute_path + '_save/Elmo-CRF/jamo_level/test_used/model/fastText_emb/'
data_save_file_path = absolute_path + '_save/Elmo-CRF/jamo_level/test_used/data/fastText_emb/'
embedding_save_file_path = absolute_path +'_save/embeddings/' + inputData + '/fastText_emb/'
model_save = model_save_file_path + inputData + "/" + inputData + '_' + str(epochs) + "_fastText_emb.json"
model_save_weights = model_save_file_path + inputData + "/" + inputData + '_' + str(epochs) + "_fastText_emb.h5"
data_save = data_save_file_path + inputData + "/" + inputData + '_' + str(epochs) + "_{0}.npy"

# 임베딩 문장 모델, text, 임베딩 vocab 저장 파일 경로
embedding_sent_text = embedding_save_file_path + 'sent_text.txt'
embedding_paths = embedding_save_file_path + 'text_embeddings.gensimmodel'
vocab_paths = embedding_save_file_path + 'text_mapping.json'

# 임베딩 형태소 태그 모델, POS_text, 임베딩 pos_vocab 저장 파일 경로
embedding_pos_text = embedding_save_file_path + 'pos_text.txt'
pos_embedding_paths = embedding_save_file_path + 'pos_embeddings.gensimmodel'
pos_vocab_paths = embedding_save_file_path + 'pos_mapping.json'

# 임베딩 jamo 모델, jamo_text, 임베딩 jamo_vocab 저장 파일 경로
embedding_jamo_text = embedding_save_file_path + 'jamo_text.txt'
jamo_embedding_paths = embedding_save_file_path + 'jamo_embeddings.gensimmodel'
jamo_vocab_paths = embedding_save_file_path + 'jamo_mapping.json'

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
    exit()

print("input data:", inputData +".txt" '\n')

train_sentences, train_sentence_post, train_sentence_ners, train_vocab, train_pos_vocab, train_ner_vocab = readData_POS.read(train_data_path+".txt")
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

# index
word_to_index, index_to_word = index.make_index(train_vocab, 'word')
pos_to_index, index_to_pos = index.make_index(train_pos_vocab, 'pos')
ner_to_index, index_to_ner = index.make_index(train_ner_vocab, 'ner')
jamo_to_index, index_to_jamo = index.make_jamo_index(train_vocab, 'jamo')

max_len = config.maxlen
max_len_char = config.max_char_len

train_sentence_text_idx = index.index_sents(train_sentences, word_to_index)
train_sentence_post_idx = index.index_sents(train_sentence_post, pos_to_index)
train_sentence_ners_idx = index.index_sents(train_sentence_ners, ner_to_index)
train_sentence_jamo_idx = index.index_jamos(train_sentences, jamo_to_index, max_len, max_len_char)
print("train_sentence_text_idx: ", train_sentence_text_idx[:2])
print("train_sentence_post_idx: ", train_sentence_post_idx[:2])
print("train_sentence_ners_idx: ", train_sentence_ners_idx[:2], '\n')
print("train_sentence_jamo_idx: ", train_sentence_jamo_idx[:2], '\n')

test_sentence_text_idx = index.index_sents(test_sentences, word_to_index)
test_sentence_post_idx = index.index_sents(test_sentence_post, pos_to_index)
test_sentence_ners_idx = index.index_sents(test_sentence_ners, ner_to_index)
test_sentence_jamo_idx = index.index_jamos(test_sentences, jamo_to_index, max_len, max_len_char)
print("test_sentence_text_idx: ", test_sentence_text_idx[:2])
print("test_sentence_post_idx: ", test_sentence_post_idx[:2])
print("test_sentence_ners_idx: ", test_sentence_ners_idx[:2], '\n')
print("test_sentence_jamo_idx: ", test_sentence_jamo_idx[:2], '\n')

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
X_train_jamos = splitPad.get_sublist(train_sentence_jamo_idx, train_idx)
X_test_jamos = splitPad.get_sublist(test_sentence_jamo_idx, test_idx)

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
elif not os.path.exists(pos_vocab_paths):
    print("no embedding pos_vocab file - Please make embedding vectors from _genEmbeddingVectors")
    exit()
elif not os.path.exists(jamo_vocab_paths):
    print("no embedding jamo_vocab file - Please make embedding vectors from _genEmbeddingVectors")
    exit()
else:
    fastText_vocab, _ = fastText.load_vocab(vocab_paths)
    fastText_model = fastText.load_model(embedding_paths)
    fastText_pvocab, _ = fastText.load_vocab(pos_vocab_paths)
    fastText_pmodel = fastText.load_model(pos_embedding_paths)
    fastText_jvocab, _ = fastText.load_vocab(jamo_vocab_paths)
    fastText_jmodel = fastText.load_model(jamo_embedding_paths)

print("------------------------------------------------------------------------------", '\n')
# embedding matrices
print("creating embedding matrices...\n")
MAX_VOCAB = len(list(word_to_index.keys()))
TAG_VOCAB = len(list(index_to_pos.keys()))
JAMO_VOCAB = len(list(index_to_jamo.keys()))
EMBEDDING_SIZE = 100 # from default gensim model, see preprocessing.ipynb
word_embedding_matrix = np.zeros((MAX_VOCAB, EMBEDDING_SIZE))

for word in word_to_index.keys():
    # get the word vector from the embedding model
    # if it's there (check against vocab list)
    if word in fastText_vocab:
        # get the word vector
        word_vector = fastText_model[word]
        # slot it in at the proper index
    word_embedding_matrix[word_to_index[word]] = word_vector

pos_embedding_matrix = np.zeros((TAG_VOCAB, EMBEDDING_SIZE))

for pos in pos_to_index.keys():
    # get the word vector from the embedding model
    # if it's there (check against vocab list)
    if pos in fastText_pvocab:
        # get the word vector
        pos_vector = fastText_pmodel[pos]
        # slot it in at the proper index
    pos_embedding_matrix[pos_to_index[pos]] = pos_vector

jamo_embedding_matrix = np.zeros((JAMO_VOCAB, EMBEDDING_SIZE))

for jamo in jamo_to_index.keys():
    # get the word vector from the embedding model
    # if it's there (check against vocab list)
    if jamo in fastText_jvocab:
        # get the word vector
        jamo_vector = fastText_jmodel[jamo]
        # slot it in at the proper index
    jamo_embedding_matrix[jamo_to_index[jamo]] = jamo_vector

print("------------------------------------------------------------------------------", '\n')


MAX_VOCAB = len(list(word_to_index.keys()))
TAG_VOCAB = len(list(index_to_pos.keys()))
NER_VOCAB = len(list(index_to_ner.keys()))
JAMO_VOCAB = len(list(index_to_jamo.keys()))
print("MAX_VOCAB:", MAX_VOCAB)
print("TAG_VOCAB:", TAG_VOCAB)
print("NER_VOCAB:", NER_VOCAB)
print("JAMO_VOCAB:", JAMO_VOCAB)


DROPOUTRATE = 0.2
cell_clip = 5
proj_clip = 5
hidden_units_size = 400
word_dropout_rate = 0.05

# model
print('Building model...\n')
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Conv1D, Flatten, BatchNormalization, Activation
from keras.layers import Bidirectional, concatenate, SpatialDropout1D, GlobalMaxPooling1D, MaxPooling1D, add, Lambda
from keras.constraints import MinMaxNorm
import numpy as np
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

def reverse(inputs, axes=1):
    return K.reverse(inputs, axes=axes)
# Conv1D	Extracts local features using 1D filters.
# 필터를 이용하여 지역적인 특징을 추출
# GlobalMaxPooling1D	Returns the largest vector of several input vectors.
# 여러 개의 벡터 정보 중 가장 큰 벡터를 골라서 반환
# MaxPooling1D	Returns the largest vectors of specific range of input vectors.
# 입력벡터에서 특정 구간마다 값을 골라 벡터를 구성한 후 반환

# text layers
txt_input = Input(shape=(max_len,), name='txt_input')
txt_embed = Embedding(MAX_VOCAB, EMBEDDING_SIZE, input_length=max_len, weights=[word_embedding_matrix],
                      name='txt_embedding', trainable=True)(txt_input)
txt_drpot = Dropout(DROPOUTRATE, name='txt_dropout')(txt_embed)
#txt_lstml = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1), name='txt_bidirectional')(txt_drpot)

# pos layers
pos_input = Input(shape=(max_len,), name='pos_input')
pos_embed = Embedding(TAG_VOCAB, EMBEDDING_SIZE, input_length=max_len, weights=[pos_embedding_matrix],
                      name='pos_embedding', trainable=True)(pos_input)
pos_drpot = Dropout(DROPOUTRATE, name='pos_dropout')(pos_embed)
#pos_lstml = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1), name='pos_bidirectional')(pos_drpot)

# jamo layers (TimeDistributed 레이어의 문자에 적용 할 부분을 wrap하여서, 모든 문자 시퀀스에 동일한 레이어를 적용.)
jamo_input = Input(shape=(max_len, max_len_char*3,), name='jamo_input')
jamo_embed = TimeDistributed(Embedding(JAMO_VOCAB, output_dim=20, input_length=max_len_char*3, weights=[jamo_embedding_matrix],
                      name='jamo_embedding', trainable=True))(jamo_input)


# character CNN to get word encodings by characters
'''
conv_layers = [[256, 7, 3],
               [256, 7, 3],
               [256, 3, -1],
               [256, 3, -1],
               [256, 3, -1],
               [256, 3, 3]]

fully_connected_layers = [1024, 1024]

for filter_num, filter_size, pooling_size in conv_layers:
    char_enc = TimeDistributed(Conv1D(filter_num, filter_size))(char_embed)
    char_enc = TimeDistributed(Activation('tanh'))(char_enc)
    if pooling_size != -1:
        char_enc = TimeDistributed(MaxPooling1D(pool_size=pooling_size))(char_enc)
char_enc = TimeDistributed(Flatten())(char_enc)

#fully connected layers
for dense_size in fully_connected_layers:
    char_enc = TimeDistributed(Dense(dense_size, activation='softmax'))(char_enc)
    char_enc = Dropout(0.5)(char_enc)

'''
# 1-size window CNN with batch-norm & tanh activation
cnns1 = TimeDistributed(Conv1D(filters=20, kernel_size=1, padding="same", strides=1), name='cnn1_cnn')(jamo_embed)
cnns1 = TimeDistributed(BatchNormalization(), name='cnn1_bnorm')(cnns1)
cnns1 = TimeDistributed(Activation('tanh'), name='cnn1_act')(cnns1)
cnns1 = TimeDistributed(GlobalMaxPooling1D(), name='cnn1_gmp')(cnns1)

# 2-size window CNN with batch-norm & tanh activation
cnns2 = TimeDistributed(Conv1D(filters=40, kernel_size=2, padding="same", strides=1), name='cnn2_cnn')(jamo_embed)
cnns2 = TimeDistributed(BatchNormalization(), name='cnn2_bnorm')(cnns2)
cnns2 = TimeDistributed(Activation('tanh'), name='cnn2_act')(cnns2)
cnns2 = TimeDistributed(GlobalMaxPooling1D(), name='cnn2_gmp')(cnns2)

# 3-size window CNN with batch-norm & tanh activation
cnns3 = TimeDistributed(Conv1D(filters=60, kernel_size=3, padding="same", strides=1), name='cnn3_cnn')(jamo_embed)
cnns3 = TimeDistributed(BatchNormalization(), name='cnn3_bnorm')(cnns3)
cnns3 = TimeDistributed(Activation('tanh'), name='cnn3_act')(cnns3)
cnns3 = TimeDistributed(GlobalMaxPooling1D(), name='cnn3_gmp')(cnns3)

# 4-size window CNN with batch-norm & tanh activation
cnns4 = TimeDistributed(Conv1D(filters=80, kernel_size=4, padding="same", strides=1), name='cnn4_cnn')(jamo_embed)
cnns4 = TimeDistributed(BatchNormalization(), name='cnn4_bnorm')(cnns4)
cnns4 = TimeDistributed(Activation('tanh'), name='cnn4_act')(cnns4)
cnns4 = TimeDistributed(GlobalMaxPooling1D(), name='cnn4_gmp')(cnns4)

cnns  = concatenate([cnns1, cnns2, cnns3, cnns4], axis=-1, name='cnn_concat')

########################################
# subword vector highway layer
########################################
import keras.backend as K
from keras.layers import Multiply, Add, Lambda
import keras.initializers

hway_input = Input(shape=(K.int_shape(cnns)[-1],))
gate_bias_init = keras.initializers.Constant(-2)
transform_gate = Dense(units=K.int_shape(cnns)[-1], bias_initializer=gate_bias_init, activation='sigmoid')(hway_input)
carry_gate = Lambda(lambda x: 1.0 - x, output_shape=(K.int_shape(cnns)[-1],))(transform_gate)
h_transformed = Dense(units=K.int_shape(cnns)[-1])(hway_input)
h_transformed = Activation('relu')(h_transformed)
transformed_gated = Multiply()([transform_gate, h_transformed])
carried_gated = Multiply()([carry_gate, hway_input])
outputs = Add()([transformed_gated, carried_gated])

highway = Model(inputs=hway_input, outputs=outputs)

cnns = TimeDistributed(highway, name='cnn_highway')(cnns)


# Concatenate inputs
model = concatenate([txt_embed, pos_embed, cnns], axis=-1) #?
#model = concatenate([txt_lstml, pos_lstml, cnns], axis=2)
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

output = concatenate([outputs, re_outputs], axis=2)

# output layer
out = TimeDistributed(Dense(50, activation='relu'))(output)  # softmax output layer
crf = CRF(NER_VOCAB)
mrg_chain = crf(out)

model = Model(inputs=[txt_input, pos_input, jamo_input], outputs=mrg_chain)
model.compile(optimizer="adam", loss=crf_loss, metrics=[crf_viterbi_accuracy])
                         #adam, rmsprop


from keras.utils import np_utils
y_train_ner = np_utils.to_categorical(y_train_ner)

epochs = config.epochs
data_name = config.data_name

model.summary()
history = model.fit([X_train_sents, X_train_pos,
                     np.array(X_train_jamos).reshape((len(X_train_jamos), max_len, max_len_char*3))],
                     y_train_ner,
                     batch_size=config.batch_size, epochs=epochs, validation_split=config.validation_rate, verbose=1)


y_test2 = np_utils.to_categorical(y_test_ner)

print("\n 테스트 정확도: %.4f" % (model.evaluate([X_test_sents, X_test_pos, np.array(X_test_jamos).reshape((len(X_test_jamos),
                                                                                          max_len, max_len_char*3))], y_test2)[1]))

print("------------------------------------------------------------------------------", '\n')
# save

print('save model & parameters.................')
save.modelsave(model, model_save, model_save_weights)

saves = [X_test_sents, X_test_pos, X_test_jamos, y_test_ner, word_to_index, ner_to_index, pos_to_index, jamo_to_index, index_to_word, index_to_ner]
names = ['X_test_sents', 'X_test_pos', 'X_test_jamos', 'y_test_ner', 'word_to_index', 'ner_to_index', 'pos_to_index', 'jamo_to_index', 'index_to_word', 'index_to_ner']

save.numpy_save(saves, names, data_save)

# evaluation

print('evaluation.................', '\n')
i=2 # 확인하고 싶은 테스트용 샘플의 인덱스.

eval.testData_POS(i, model, X_test_sents, X_test_pos, X_test_jamos, max_len, max_len_char, y_test2, index_to_word, index_to_ner)

y_predicted = model.predict([X_test_sents, X_test_pos, np.array(X_test_jamos).reshape((len(X_test_jamos), max_len, max_len_char*3))])
pred_tags = eval.sequences_to_tag(y_predicted, index_to_ner)
test_tags = eval.sequences_to_tag(y_test2, index_to_ner)

from _utils.common.classification_report import classification_report
from _utils.gru.utils import classification_report_to_txt

tmp_accuracy = model.evaluate([X_test_sents, X_test_pos, np.array(X_test_jamos).reshape((len(X_test_jamos),
                                                                                          max_len, max_len_char*3))], y_test2)[1]

report, accuracy = classification_report(test_tags, pred_tags, tmp_accuracy)
print("\n", report)
classification_report_to_txt(config, accuracy, "fastText", "../result/", report)