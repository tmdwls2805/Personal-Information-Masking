import os
from _utils.common import readData_POS
from _embeddingModels.elmo import sent_utils, train_elmo
from _utils.common.config import Config

config = Config(json_path="./config.json")

absolute_path = config.absolute_path
input_path = absolute_path + config.data_dir

inputData = config.training_data
train_data_path = input_path + inputData + '.txt'

embedding_save_path = absolute_path + '_save/embeddings/' + inputData + '/elmo_emb/'

pretrain_ckpt_word_file_path = embedding_save_path + 'pretrain_ckpt/words/'
embedding_word_weight_paths = pretrain_ckpt_word_file_path + 'word_weights.hdf5'

embedding_sent_text = embedding_save_path + 'sent_text.txt'
elmo_vocab = pretrain_ckpt_word_file_path + 'elmo-vocab.txt'

if not os.path.exists(embedding_save_path):
    print("embedding_save_path is not exist")
    os.makedirs(embedding_save_path)
    exit()

if not os.path.exists(pretrain_ckpt_word_file_path):
    print("pretrain_ckpt_word_file_path is not exist")
    os.makedirs(pretrain_ckpt_word_file_path)
    exit()


train_sentences, train_sentence_post, train_sentence_ners, train_vocab, train_pos_vocab, train_ner_vocab = readData_POS.read(train_data_path)


with open(embedding_sent_text, 'w', encoding='cp949') as f:
     for s in train_sentences:
         f.write(' '.join(s))
         f.write('\n')

token_num = sent_utils.construct_elmo_vocab(embedding_sent_text, elmo_vocab)

n_gpus = 1

train_elmo.main(embedding_sent_text, elmo_vocab, pretrain_ckpt_word_file_path, token_num, n_gpus)
