import os
from _utils.common import readData_POS
from _utils.jamo import split_char
from _utils.jamo import jamo_split
from _embeddingModels.glove import glove
from _utils.common.config import Config

config = Config(json_path="./config.json")

absolute_path = config.absolute_path
input_path = absolute_path + config.data_dir

inputData = config.training_data
train_data_path = input_path + inputData + '.txt'

embedding_save_path = absolute_path + '_save/embeddings/' + inputData + '/glove_emb/'

# 임베딩 문장 모델, text, 임베딩 vocab 저장 파일 경로
embedding_sent_text = embedding_save_path + 'sent_text.txt'
embedding_paths = embedding_save_path + 'text_embeddings.gensimmodel'
vocab_paths = embedding_save_path + 'text_mapping.json'

# 임베딩 형태소 태그 모델, POS_text, 임베딩 pos_vocab 저장 파일 경로
embedding_pos_text = embedding_save_path + 'pos_text.txt'
pos_embedding_paths = embedding_save_path + 'pos_embeddings.gensimmodel'
pos_vocab_paths = embedding_save_path + 'pos_mapping.json'

# 임베딩 Character 태그 모델, char_text, 임베딩 char_vocab 저장 파일 경로
embedding_char_text = embedding_save_path + 'char_text.txt'
char_embedding_paths = embedding_save_path + 'char_embeddings.gensimmodel'
char_vocab_paths = embedding_save_path + 'char_mapping.json'

# 임베딩 jamo 태그 모델, jamo_text, 임베딩 jamo_vocab 저장 파일 경로
embedding_jamo_text = embedding_save_path + 'jamo_text.txt'
jamo_embedding_paths = embedding_save_path + 'jamo_embeddings.gensimmodel'
jamo_vocab_paths = embedding_save_path + 'jamo_mapping.json'

if not os.path.exists(embedding_save_path):
    print("embedding_save_file_path is not exist")
    os.makedirs(embedding_save_path)
    exit()

# 학습 데이터 내 각 문장, 학습 데이터 내 각 문장에서 사용된 형태소 태그, 학습 데이터 내 각 문장에서 사용된 개체명,
# 학습 데이터 단어 집합, 학습 데이터 형태소 태그 집합, 학습 데이터 개체명 태그 집합
train_sentences, train_sentence_post, train_sentence_ners, train_vocab, train_pos_vocab, train_ner_vocab = readData_POS.read(train_data_path)

# sentence embeddings
with open(embedding_sent_text, 'w', encoding='utf-8') as f:
    for s in train_sentences:
        f.write(' '.join(s))
        f.write('\n')

glove_vocab, glove_model = glove.create_embeddings(embedding_sent_text,
                       embeddings_path=embedding_paths,
                       vocab_path=vocab_paths,
                       window=3,
                       min_count=3,
                       workers=4,
                       iter=20)

# pos embeddings
with open(embedding_pos_text, 'w', encoding='utf-8') as f:
    for s in train_sentence_post:
        f.write(' '.join(s))
        f.write('\n')

glove_pvocab, glove_pmodel = glove.create_embeddings(embedding_pos_text,
                       embeddings_path=pos_embedding_paths,
                       vocab_path=pos_vocab_paths,
                       window=3,
                       min_count=3,
                       workers=4,
                       iter=20)

# char embeddings
with open(embedding_char_text, 'w', encoding='utf-8') as f:
    for s in train_sentences:
        chars = [w_i for w in s for w_i in w]
        f.write(' '.join(chars))
        f.write('\n')

glove_cvocab, glove_cmodel = glove.create_embeddings(embedding_char_text,
                       embeddings_path=char_embedding_paths,
                       vocab_path=char_vocab_paths,
                       window=3,
                       min_count=3,
                       workers=4,
                       iter=20)

# jamo embeddings
with open(embedding_jamo_text, 'w', encoding='utf-8') as f:
    for s in train_sentences:
        chars = [w_i for w in s for w_i in w]
        vocab_character = []
        for ch in chars:
            vocab_character.append(' '.join(split_char.split_char_convert(ch)))
        f.write(' '.join(vocab_character))
        f.write('\n')

glove_jvocab, glove_jmodel = glove.create_embeddings(embedding_jamo_text,
                       embeddings_path=jamo_embedding_paths,
                       vocab_path=jamo_vocab_paths,
                       window=3,
                       min_count=3,
                       workers=4,
                       iter=20)
