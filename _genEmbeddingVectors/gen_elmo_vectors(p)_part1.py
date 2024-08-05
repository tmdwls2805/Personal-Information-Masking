import os
from _utils.common import readData_POS
from _embeddingModels.elmo import sent_utils, train_elmo
from _utils.common.config import Config

config = Config(json_path="./config.json")

absolute_path = config.absolute_path
input_path = absolute_path + config.data_dir

inputData = config.training_data
train_data_path = input_path + inputData +'.txt'

embedding_save_file_path = absolute_path + '_save/embeddings/' + inputData + '/elmo_emb/'

pretrain_ckpt_pos_file_path = embedding_save_file_path + 'pretrain_ckpt/pos/'


# 임베딩 문장 모델, text, 임베딩 vocab 저장 파일 경로
embedding_pos_text = embedding_save_file_path + 'pos_text.txt'
elmo_pos = pretrain_ckpt_pos_file_path  + 'elmo-pos.txt'

if not os.path.exists(embedding_save_file_path):
    print("embedding_save_file_path is not exist")
    os.makedirs(embedding_save_file_path)
    exit()

if not os.path.exists(pretrain_ckpt_pos_file_path):
    print("pretrain_ckpt_pos_file_path is not exist")
    os.makedirs(pretrain_ckpt_pos_file_path)
    exit()

# 학습 데이터 내 각 문장, 학습 데이터 내 각 문장에서 사용된 형태소 태그, 학습 데이터 내 각 문장에서 사용된 개체명,
# 학습 데이터 단어 집합, 학습 데이터 형태소 태그 집합, 학습 데이터 개체명 태그 집합
train_sentences, train_sentence_post, train_sentence_ners, train_vocab, train_pos_vocab, train_ner_vocab = readData_POS.read(train_data_path)


# sentence embeddings
with open(embedding_pos_text, 'w', encoding='cp949') as f:
    for s in train_sentence_post:
        f.write(' '.join(s))
        f.write('\n')

token_num = sent_utils.construct_elmo_vocab(embedding_pos_text, elmo_pos)

n_gpus = 1  # Number of GPU (GPU 개수가 다르면 수정 필요)

train_elmo.main(embedding_pos_text, elmo_pos, pretrain_ckpt_pos_file_path, token_num, n_gpus)

