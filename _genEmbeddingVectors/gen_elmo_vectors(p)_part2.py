from _embeddingModels.elmo.bilm import dump_weights as dump_elmo_weights
from _utils.common.config import Config

config = Config(json_path="./config.json")

absolute_path = config.absolute_path
input_path = absolute_path + config.data_dir

inputData = config.training_data
train_data_path = input_path + inputData + '.txt'

embedding_save_file_path = absolute_path + '_save/embeddings/' + inputData + '/elmo_emb/'
pretrain_ckpt_pos_file_path = embedding_save_file_path + 'pretrain_ckpt/pos/'
embedding_pos_weight_paths = pretrain_ckpt_pos_file_path + 'pos_weights.hdf5'

dump_elmo_weights(pretrain_ckpt_pos_file_path, embedding_pos_weight_paths)