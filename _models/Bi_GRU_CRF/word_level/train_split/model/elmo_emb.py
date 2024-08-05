import os
from _utils.word import index, eval
from _utils.common import readData_POS, splitPad, save
from _utils.common.config import Config
from sklearn.model_selection import train_test_split
import numpy as np
import sys

config = Config(json_path="../config.json")

absolute_path = config.absolute_path
input_path = absolute_path + config.data_dir

# 입력 데이터 이름 (_inputData 폴더)
inputData = config.training_data
epochs = config.epochs
data_name = config.data_name

train_data_path = input_path + inputData

# 모델, weight, 데이터 저장 파일 경로
model_save_path = absolute_path + '_save/Bi-GRU-CRF/word_level/train_split/model/elmo_emb/'
data_save_path = absolute_path + '_save/Bi-GRU-CRF/word_level/train_split/data/elmo_emb/'
embedding_save_path = absolute_path + '_save/embeddings/' + inputData + '/elmo_emb/'
model_save = model_save_path + inputData + "/" + inputData + '_' + str(epochs) + "_elmo_emb.json"
model_save_weights = model_save_path + inputData + "/" + inputData + '_' + str(epochs) + "_elmo_emb.h5"
data_save = data_save_path + inputData + "/" + inputData + '_' + str(epochs) + "_{0}.npy"
pretrain_ckpt_word_file_path = embedding_save_path + 'pretrain_ckpt/words/'
pretrain_ckpt_pos_file_path = embedding_save_path + 'pretrain_ckpt/pos/'

# 임베딩 문장 모델, text, 임베딩 vocab 저장 파일 경로
embedding_sent_text = embedding_save_path + 'sent_text.txt'
elmo_vocab = pretrain_ckpt_word_file_path + 'elmo-vocab.txt'
embedding_weight_paths = pretrain_ckpt_word_file_path + 'word_weights.hdf5'

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
from allennlp.commands.elmo import ElmoEmbedder

elmo_model = ElmoEmbedder(options_file=pretrain_ckpt_word_file_path + "options.json", weight_file=embedding_weight_paths)
elmo_pos_model = ElmoEmbedder(options_file=pretrain_ckpt_pos_file_path  + "options.json", weight_file=embedding_pos_weight_paths)

# Embedding matrices
print("Creating embedding matrices...\n")
MAX_VOCAB = len(list(word_to_index.keys()))
POS_VOCAB = len(list(index_to_pos.keys()))
EMBEDDING_SIZE = 256
word_embedding_matrix = np.zeros((MAX_VOCAB, EMBEDDING_SIZE))
pos_embedding_matrix = np.zeros((POS_VOCAB, EMBEDDING_SIZE))

wordIdx = 0
for word in word_to_index.keys():
    # get the word vector
    elmo_embedding = elmo_model.embed_sentence(word)
    avg_elmo_embedding = np.average(elmo_embedding, axis=0)
    word_embedding_matrix[word_to_index[word]] = avg_elmo_embedding[0]

    # some progress info
    wordIdx += 1
    percent = 100.0 * wordIdx / word_to_index.__len__()
    line = '[{0}{1}]'.format('=' * int(percent / 2), ' ' * (50 - int(percent / 2)))
    status = '\r{0:3.0f}%{1} {2:3d}/{3:3d} words'
    sys.stdout.write(status.format(percent, line, wordIdx, word_to_index.__len__()))


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


MAX_VOCAB = len(list(word_to_index.keys()))
POS_VOCAB = len(list(index_to_pos.keys()))
NER_VOCAB = len(list(index_to_ner.keys()))
print("MAX_VOCAB:", MAX_VOCAB)
print("TAG_VOCAB:", POS_VOCAB)
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
                      name='txt_embedding', trainable=True)(txt_input)
txt_drpot = Dropout(DROPOUTRATE, name='txt_dropout')(txt_embed)


# pos layers
pos_input = Input(shape=(max_len,), name='pos_input')
pos_embed = Embedding(POS_VOCAB, EMBEDDING_SIZE, input_length=max_len,
                      weights=[pos_embedding_matrix],
                      name='pos_embedding', trainable=True)(pos_input)
pos_drpot = Dropout(DROPOUTRATE, name='pos_dropout')(pos_embed)


# Concatenate input
model = concatenate([txt_drpot, pos_drpot], axis=2)
model = SpatialDropout1D(DROPOUTRATE)(model)

# Deep Layers
model = Bidirectional(GRU(units=100, return_sequences=True, recurrent_dropout=0.1))(model)

# Output layers
out = TimeDistributed(Dense(50, activation='relu'))(model)
crf = CRF(NER_VOCAB)
mrg_chain = crf(out)

model = Model(inputs=[txt_input, pos_input], outputs=mrg_chain)
model.compile(optimizer='rmsprop', loss=crf_loss, metrics=[crf_viterbi_accuracy])
                        #adam



from keras.utils import np_utils
y_train2 = np_utils.to_categorical(y_train_ner)

model.summary()
history = model.fit([X_train_sents, X_train_pos], y_train2, batch_size=32, epochs=epochs, validation_split=0.1, verbose=1)

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

report = classification_report(test_tags, pred_tags)
print("\n", report)
classification_report_to_txt(epochs, "elmo", data_name, "../result/", report)
