import os
from _utils.word import index, eval
from _utils.common import readData_POS, splitPad, save
from _utils.common.config import Config
import numpy as np
import sys

config = Config(json_path="../config.json")

absolute_path = config.absolute_path
input_path = absolute_path + config.data_dir

# 입력 데이터 이름 (_inputData 폴더)
inputData = config.training_data
train_data_path = input_path + inputData
epochs = config.epochs

# 테스트 데이터 이름 (_inputData 폴더)
testData = config.test_data
test_data_path = input_path + testData

# 모델, weight, 데이터 저장 파일 경로
model_save_path = absolute_path + '_save/Bi-GRU-CRF/word_level/test_used/model/elmo_emb/'
data_save_path = absolute_path + '_save/Bi-GRU-CRF/word_level/test_used/data/elmo_emb/'
embedding_save_path = absolute_path + '_save/embeddings/' + inputData + '/elmo_emb/'
model_save = model_save_path + inputData + "/" + inputData + '_' + str(epochs) + "_elmo_emb.json"
model_save_weights = model_save_path + inputData + "/" + inputData + '_' + str(epochs) + "_elmo_emb.h5"
data_save = data_save_path + inputData + "/" + inputData + '_' + str(epochs) + "_{0}.npy"
pretrain_ckpt_word_file_path = embedding_save_path + 'pretrain_ckpt/words/'
pretrain_ckpt_pos_file_path = embedding_save_path + 'pretrain_ckpt/pos/'

# 임베딩 문장 모델, text, 임베딩 vocab 저장 파일 경로
embedding_sent_text = embedding_save_path + 'sent_text.txt'
elmo_vocab = pretrain_ckpt_word_file_path + 'elmo-vocab.txt'
embedding_word_weight_paths = pretrain_ckpt_word_file_path + 'word_weights.hdf5'

# 임베딩 형태소 태그 모델, POS_text, 임베딩 pos_vocab 저장 파일 경로
embedding_pos_text = embedding_save_path + 'pos_text.txt'
elmo_pos = pretrain_ckpt_pos_file_path + 'elmo-pos.txt'
embedding_pos_weight_paths = pretrain_ckpt_pos_file_path + 'pos_weights.hdf5'

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

print("input data:", inputData + ".txt" '\n')
print("------------------------------------------------------------------------------", '\n')

# 학습 데이터 내 각 문장, 학습 데이터 내 각 문장에서 사용된 형태소 태그, 학습 데이터 내 각 문장에서 사용된 개체명,
# 학습 데이터 단어 집합, 학습 데이터 형태소 태그 집합, 학습 데이터 개체명 태그 집합
train_sentences, train_sentence_post, train_sentence_ners, train_vocab, train_pos_vocab, train_ner_vocab = readData_POS.read(
    train_data_path+".txt")

# 테스트 데이터 내 각 문장, 테스트 데이터 내 각 문장에서 사용된 형태소 태그, 테스트 데이터 내 각 문장에서 사용된 개체명,
# 테스트 데이터 단어 집합, 테스트 데이터 형태소 태그 집합, 테스트 데이터 개체명 태그 집합
test_sentences, test_sentence_post, test_sentence_ners, test_vocab, test_pos_vocab, test_ner_vocab = readData_POS.read(
    test_data_path+".txt")

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
from allennlp.commands.elmo import ElmoEmbedder

# glove_vocab = glove.load_vocab(vocab_paths)
elmo_word_model = ElmoEmbedder(options_file=pretrain_ckpt_word_file_path  + "options.json", weight_file=embedding_word_weight_paths)
elmo_pos_model = ElmoEmbedder(options_file=pretrain_ckpt_pos_file_path  + "options.json", weight_file=embedding_pos_weight_paths)

print("------------------------------------------------------------------------------", '\n')

# embedding matrices
print("creating embedding matrices...\n")
MAX_VOCAB = len(list(word_to_index.keys()))
TAG_VOCAB = len(list(index_to_pos.keys()))
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

print("------------------------------------------------------------------------------", '\n')

MAX_VOCAB = len(list(word_to_index.keys()))
TAG_VOCAB = len(list(index_to_pos.keys()))
NER_VOCAB = len(list(index_to_ner.keys()))
print("MAX_VOCAB:", MAX_VOCAB)
print("TAG_VOCAB:", TAG_VOCAB)
print("NER_VOCAB:", NER_VOCAB)


DROPOUTRATE = 0.2
# model
print('Building model...\n')
from keras.models import Model, Input
from keras.layers import concatenate, GRU, LSTM, Embedding, Dense, TimeDistributed, Bidirectional, Dropout, SpatialDropout1D
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

# text layers
txt_input = Input(shape=(max_len,), name='txt_input')
txt_embed = Embedding(MAX_VOCAB, EMBEDDING_SIZE, input_length=max_len,
                      weights=[word_embedding_matrix],
                      name='txt_embedding', trainable=True, mask_zero=True)(txt_input)
txt_drpot = Dropout(DROPOUTRATE, name='txt_dropout')(txt_embed)

# pos layers
pos_input = Input(shape=(max_len,), name='pos_input')
pos_embed = Embedding(TAG_VOCAB, EMBEDDING_SIZE, input_length=max_len,
                      weights=[pos_embedding_matrix],
                      name='pos_embedding', trainable=True)(pos_input)
pos_drpot = Dropout(DROPOUTRATE, name='pos_dropout')(pos_embed)

model = concatenate([txt_drpot, pos_drpot], axis=2)
model = SpatialDropout1D(0.3)(model)

# Deep Layers
model = Bidirectional(GRU(units=100, return_sequences=True, recurrent_dropout=0.1))(model)

# output layer
out = TimeDistributed(Dense(50, activation='relu'))(model)  # softmax output layer
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
classification_report_to_txt(config, accuracy, "elmo", "../result/", report)