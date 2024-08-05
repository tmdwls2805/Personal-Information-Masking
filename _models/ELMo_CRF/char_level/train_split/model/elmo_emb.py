# Char-Level, Bi-LSTM, elmo_emb POS 사용 모델
# Word, POS Tag, char 모두 elmo embedding 수행한 모델

import os
from _utils.char import index, eval
from _utils.common import readData_POS, splitPad, save
from _utils.common.config import Config
from sklearn.model_selection import train_test_split
import numpy as np
import sys

config = Config(json_path="../config.json")

absolute_path = config.absolute_path
input_path = absolute_path + config.data_dir
epochs = config.epochs

data_name = config.data_name

# 입력 데이터 이름 (_inputData 폴더)
inputData = config.training_data
train_data_path = input_path + inputData

# 모델, weight, 데이터 저장 파일 경로
model_save_file_path = absolute_path + '_save/Elmo-CRF/char_level/train_split/model/elmo_emb/'
data_save_file_path = absolute_path + '_save/Elmo-CRF/char_level/train_split/data/elmo_emb/'
embedding_save_file_path = absolute_path + '_save/embeddings/' + inputData + '/elmo_emb/'
model_save = model_save_file_path + inputData + "/" + inputData + '_' + str(epochs) + "_elmo_emb.json"
model_save_weights = model_save_file_path + inputData + "/" + inputData + '_' + str(epochs) + "_elmo_emb.h5"
data_save = data_save_file_path + inputData + "/" + inputData + '_' + str(epochs) + "_{0}.npy"
pretrain_ckpt_word_file_path = embedding_save_file_path + 'pretrain_ckpt/words/'
pretrain_ckpt_pos_file_path = embedding_save_file_path + 'pretrain_ckpt/pos/'
pretrain_ckpt_char_file_path = embedding_save_file_path + 'pretrain_ckpt/char/'

# 임베딩 문장 모델, text, 임베딩 vocab 저장 파일 경로
embedding_sent_text = embedding_save_file_path + 'sent_text.txt'
elmo_vocab = pretrain_ckpt_word_file_path + 'elmo-vocab.txt'
embedding_word_weight_paths = pretrain_ckpt_word_file_path + 'word_weights.hdf5'

# 임베딩 형태소 태그 모델, POS_text, 임베딩 pos_vocab 저장 파일 경로
embedding_pos_text = embedding_save_file_path + 'pos_text.txt'
elmo_pos = pretrain_ckpt_pos_file_path + 'elmo-pos.txt'
embedding_pos_weight_paths = pretrain_ckpt_pos_file_path + 'pos_weights.hdf5'

# 임베딩 character 태그 모델, char_text, 임베딩 char_vocab 저장 파일 경로
embedding_char_text = embedding_save_file_path + 'char_text.txt'
elmo_char = pretrain_ckpt_char_file_path + 'elmo-char.txt'
embedding_char_weight_paths = pretrain_ckpt_char_file_path + 'char_weights.hdf5'

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
print("------------------------------------------------------------------------------", '\n')

# 학습 데이터 내 각 문장, 학습 데이터 내 각 문장에서 사용된 형태소 태그, 학습 데이터 내 각 문장에서 사용된 개체명,
sentences, sentence_post, sentence_ners, vocab, pos_vocab, ner_vocab = readData_POS.read(train_data_path+".txt")

print("총 단어개수: ", len(vocab))
print("sentences (대표 3개만 보기): ", sentences[:3])
print("Vocab(단어 집합)", vocab)
print("POS 태그: ", pos_vocab)
print("NER 태그: ", ner_vocab, '\n')

print("------------------------------------------------------------------------------", '\n')

max_len = config.maxlen
max_len_char = config.max_char_len
# index
word_to_index, index_to_word = index.make_index(vocab, 'word')
pos_to_index, index_to_pos = index.make_index(pos_vocab, 'pos')
ner_to_index, index_to_ner = index.make_index(ner_vocab, 'ner')
char_to_index, index_to_char = index.make_char_index(vocab, 'char')

sentence_text_idx = index.index_sents(sentences, word_to_index)
sentence_post_idx = index.index_sents(sentence_post, pos_to_index)
sentence_ners_idx = index.index_sents(sentence_ners, ner_to_index)
sentence_char_idx = index.index_chars(sentences, char_to_index, max_len, max_len_char)
print("train_sentence_text_idx: ", sentence_text_idx[:2])
print("train_sentence_post_idx: ", sentence_post_idx[:2])
print("train_sentence_ners_idx: ", sentence_ners_idx[:2], '\n')
print("train_sentence_char_idx: ", sentence_char_idx[:2], '\n')

print("------------------------------------------------------------------------------", '\n')

# Split train & test sets, Pad data
indices = [i for i in range(len(sentences))]
train_idx, test_idx, X_train_pos, X_test_pos = train_test_split(indices, sentence_post_idx, test_size=config.test_rate, random_state=1)


X_train_sents = splitPad.get_sublist(sentence_text_idx, train_idx)
X_test_sents = splitPad.get_sublist(sentence_text_idx, test_idx)
y_train_ner = splitPad.get_sublist(sentence_ners_idx, train_idx)
y_test_ner = splitPad.get_sublist(sentence_ners_idx, test_idx)
X_train_chars = splitPad.get_sublist(sentence_char_idx, train_idx)
X_test_chars = splitPad.get_sublist(sentence_char_idx, test_idx)

max_len = config.maxlen
X_train_sents = splitPad.pad_data(X_train_sents, max_len)
X_test_sents = splitPad.pad_data(X_test_sents, max_len)
X_train_pos = splitPad.pad_data(X_train_pos, max_len)
X_test_pos = splitPad.pad_data(X_test_pos, max_len)
y_train_ner = splitPad.pad_data(y_train_ner, max_len)
y_test_ner = splitPad.pad_data(y_test_ner, max_len)

print("------------------------------------------------------------------------------", '\n')

# load embedding data
from allennlp.commands.elmo import ElmoEmbedder

elmo_word_model = ElmoEmbedder(options_file=pretrain_ckpt_word_file_path  + "options.json", weight_file=embedding_word_weight_paths)
elmo_pos_model = ElmoEmbedder(options_file=pretrain_ckpt_pos_file_path  + "options.json", weight_file=embedding_pos_weight_paths)
elmo_char_model = ElmoEmbedder(options_file=pretrain_ckpt_char_file_path  + "options.json", weight_file=embedding_char_weight_paths)


print("------------------------------------------------------------------------------", '\n')
# embedding matrices
print("creating embedding matrices...\n")
MAX_VOCAB = len(list(word_to_index.keys()))
TAG_VOCAB = len(list(index_to_pos.keys()))
CHAR_VOCAB = len(list(index_to_char.keys()))
EMBEDDING_SIZE = 256 # from default gensim model, see preprocessing.ipynb
word_embedding_matrix = np.zeros((MAX_VOCAB, EMBEDDING_SIZE))

wordIdx = 0
for word in word_to_index.keys():
    # get the word vector
    elmo_embedding = elmo_word_model.embed_sentence(word)
    avg_elmo_embedding = np.average(elmo_embedding, axis=0)
    word_embedding_matrix[word_to_index[word]] = avg_elmo_embedding[0]

    # Some progress info
    wordIdx += 1
    percent = 100.0 * wordIdx / word_to_index.__len__()
    line = '[{0}{1}]'.format('=' * int(percent / 2), ' ' * (50 - int(percent / 2)))
    status = '\r{0:3.0f}%{1} {2:3d}/{3:3d} words'
    sys.stdout.write(status.format(percent, line, wordIdx, word_to_index.__len__()))

pos_embedding_matrix = np.zeros((TAG_VOCAB, EMBEDDING_SIZE))

posIdx = 0
for pos in pos_to_index.keys():
    # get the word vector
    elmo_pos_embedding = elmo_pos_model.embed_sentence(pos)
    avg_elmo_pos_embedding = np.average(elmo_pos_embedding, axis=0)
    pos_embedding_matrix[pos_to_index[pos]] = avg_elmo_pos_embedding[0]

    # Some progress info
    posIdx += 1
    percent = 100.0 * posIdx / pos_to_index.__len__()
    line = '[{0}{1}]'.format('=' * int(percent / 2), ' ' * (50 - int(percent / 2)))
    status = '\r{0:3.0f}%{1} {2:3d}/{3:3d} pos_tags'
    sys.stdout.write(status.format(percent, line, posIdx, pos_to_index.__len__()))

char_embedding_matrix = np.zeros((CHAR_VOCAB, EMBEDDING_SIZE))

charIdx = 0
for char in char_to_index.keys():
    # get the word vector
    elmo_char_embedding = elmo_char_model.embed_sentence(char)
    avg_elmo_char_embedding = np.average(elmo_char_embedding, axis=0)
    char_embedding_matrix[char_to_index[char]] = avg_elmo_char_embedding[0]

    # Some progress info
    charIdx += 1
    percent = 100.0 * charIdx / char_to_index.__len__()
    line = '[{0}{1}]'.format('=' * int(percent / 2), ' ' * (50 - int(percent / 2)))
    status = '\r{0:3.0f}%{1} {2:3d}/{3:3d} char_tags'
    sys.stdout.write(status.format(percent, line, charIdx, char_to_index.__len__()))

print("------------------------------------------------------------------------------", '\n')


MAX_VOCAB = len(list(word_to_index.keys()))
TAG_VOCAB = len(list(index_to_pos.keys()))
NER_VOCAB = len(list(index_to_ner.keys()))
CHAR_VOCAB = len(list(index_to_char.keys()))
print("MAX_VOCAB:", MAX_VOCAB)
print("TAG_VOCAB:", TAG_VOCAB)
print("NER_VOCAB:", NER_VOCAB)
print("CHAR_VOCAB:", CHAR_VOCAB)

DROPOUTRATE = 0.2
cell_clip = 5
proj_clip = 5
hidden_units_size = 712
word_dropout_rate = 0.05

# model
print('Building model...\n')
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Conv1D, BatchNormalization, Activation, add
from keras.layers import concatenate, SpatialDropout1D, GlobalMaxPooling1D
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

# char layers (TimeDistributed 레이어의 문자에 적용 할 부분을 wrap하여서, 모든 문자 시퀀스에 동일한 레이어를 적용.)
char_input = Input(shape=(max_len, max_len_char,), name='char_input')
char_embed = TimeDistributed(Embedding(CHAR_VOCAB, EMBEDDING_SIZE, input_length=max_len_char, weights=[char_embedding_matrix],
                      name='char_embedding', trainable=True))(char_input)

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
cnns1 = TimeDistributed(Conv1D(filters=20, kernel_size=1, padding="same", strides=1), name='cnn1_cnn')(char_embed)
cnns1 = TimeDistributed(BatchNormalization(), name='cnn1_bnorm')(cnns1)
cnns1 = TimeDistributed(Activation('tanh'), name='cnn1_act')(cnns1)
cnns1 = TimeDistributed(GlobalMaxPooling1D(), name='cnn1_gmp')(cnns1)

# 2-size window CNN with batch-norm & tanh activation
cnns2 = TimeDistributed(Conv1D(filters=40, kernel_size=2, padding="same", strides=1), name='cnn2_cnn')(char_embed)
cnns2 = TimeDistributed(BatchNormalization(), name='cnn2_bnorm')(cnns2)
cnns2 = TimeDistributed(Activation('tanh'), name='cnn2_act')(cnns2)
cnns2 = TimeDistributed(GlobalMaxPooling1D(), name='cnn2_gmp')(cnns2)

# 3-size window CNN with batch-norm & tanh activation
cnns3 = TimeDistributed(Conv1D(filters=60, kernel_size=3, padding="same", strides=1), name='cnn3_cnn')(char_embed)
cnns3 = TimeDistributed(BatchNormalization(), name='cnn3_bnorm')(cnns3)
cnns3 = TimeDistributed(Activation('tanh'), name='cnn3_act')(cnns3)
cnns3 = TimeDistributed(GlobalMaxPooling1D(), name='cnn3_gmp')(cnns3)

# 4-size window CNN with batch-norm & tanh activation
cnns4 = TimeDistributed(Conv1D(filters=80, kernel_size=4, padding="same", strides=1), name='cnn4_cnn')(char_embed)
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
model = concatenate([txt_embed, pos_embed, cnns], axis=-1)
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

model = Model(inputs=[txt_input, pos_input, char_input], outputs=mrg_chain)
model.compile(optimizer="adam", loss=crf_loss, metrics=[crf_viterbi_accuracy])
                         #adam, rmsprop


from keras.utils import np_utils
y_train_ner = np_utils.to_categorical(y_train_ner)

model.summary()
history = model.fit([X_train_sents, X_train_pos, np.array(X_train_chars).reshape((len(X_train_chars), max_len, max_len_char))],
                    y_train_ner, batch_size=config.batch_size, epochs=epochs, validation_split=config.validation_rate, verbose=1)

y_test2 = np_utils.to_categorical(y_test_ner)

print("\n 테스트 정확도: %.4f" % (model.evaluate([X_test_sents, X_test_pos, np.array(X_test_chars).reshape((len(X_test_chars),
                                                                                          max_len, max_len_char))], y_test2)[1]))

print("------------------------------------------------------------------------------", '\n')
# save

print('save model & parameters.................')
save.modelsave(model, model_save, model_save_weights)

saves = [X_test_sents, X_test_pos, X_test_chars, y_test_ner, word_to_index, ner_to_index, pos_to_index, char_to_index, index_to_word, index_to_ner]
names = ['X_test_sents', 'X_test_pos', 'X_test_chars', 'y_test_ner', 'word_to_index', 'ner_to_index', 'pos_to_index', 'char_to_index', 'index_to_word', 'index_to_ner']

save.numpy_save(saves, names, data_save)

# evaluation

print('evaluation.................', '\n')
i=2 # 확인하고 싶은 테스트용 샘플의 인덱스.

eval.testData_POS(i, model, X_test_sents, X_test_pos, X_test_chars, max_len, max_len_char, y_test2, index_to_word, index_to_ner)

y_predicted = model.predict([X_test_sents, X_test_pos, np.array(X_test_chars).reshape((len(X_test_chars), max_len, max_len_char))])
pred_tags = eval.sequences_to_tag(y_predicted, index_to_ner)
test_tags = eval.sequences_to_tag(y_test2, index_to_ner)

from _utils.common.classification_report import classification_report
from _utils.gru.utils import classification_report_to_txt

tmp_accuracy = model.evaluate([X_test_sents, X_test_pos, np.array(X_test_chars).reshape((len(X_test_chars),
                                                                                          max_len, max_len_char))], y_test2)[1]

report, accuracy = classification_report(test_tags, pred_tags, tmp_accuracy)
print("\n", report)
classification_report_to_txt(config, accuracy, "elmo", "../result/", report)