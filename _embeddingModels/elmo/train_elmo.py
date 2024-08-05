from .bilm.training import train, load_vocab
from .bilm.data import BidirectionalLMDataset
from .bilm import dump_weights as dump_elmo_weights

def main(train_prefix, vocab_path, save_path, token_num, num_gpus):
    # load the vocab
    vocab = load_vocab(vocab_path, 50) #max word length길이가 50임
    print(vocab.size)

    # define the options
    batch_size = 32  # batch size for each GPU
    n_gpus = num_gpus # Number of GPU (GPU 개수가 다르면 수정 필요)

    # number of tokens in training data (this for 1B Word Benchmark)
    n_train_tokens = token_num

    prefix = train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                  shuffle_on_load=True)
    tf_save_dir = save_path
    tf_log_dir = save_path

    options = {
        'bidirectional': True,

        'char_cnn': {'activation': 'relu',
                     'embedding': {'dim': 16},
                     'filters': [[1, 32],
                                 [2, 32],
                                 [3, 64],
                                 [4, 128],
                                 [5, 256],
                                 [6, 512],
                                 [7, 1024]],
                     'max_characters_per_token': 50,
                     'n_characters': 261,
                     'n_highway': 2},

        'dropout': 0.1,

        'lstm': {
            'cell_clip': 3,
            'dim': 1024,
            'n_layers': 2,
            'proj_clip': 3,
            'projection_dim': 128,
            'use_skip_connections': True},

        'all_clip_norm_val': 10.0,

        'n_epochs': 10,
        'n_train_tokens': n_train_tokens,
        'batch_size': batch_size,
        'n_tokens_vocab': vocab.size,
        'unroll_steps': 20,
        'n_negative_samples_batch': 128,
    }
    train(options, data, int(n_gpus), tf_save_dir, tf_log_dir)
    #dump_elmo_weights(save_path, save_path + 'weights.hdf5')
