from _models.BERT_CRF.test_used.predict import bert_crf_pred
from _models.DistilKoBERT_CRF.test_used.predict import distilkobert_crf_pred
from _models.KoBERT.test_used.predict import kobert_pred
from _models.KoBERT_BiGRU_CRF.test_used.predict import kobert_bigru_crf_pred
from _models.KoBERT_BiLSTM_CRF.test_used.predict import kobert_bilstm_crf_pred
from _models.KoBERT_CRF.test_used.predict import kobert_crf_pred
from _models.KoELECTRA.test_used.predict import koelectra_pred
from _models.KoELECTRA_CRF.test_used.predict import koelectra_crf_pred
from _models.Bi_GRU_CRF.word_level.test_used.predict.query import bigru_crf_word_pred
from _models.Bi_GRU_CRF.jamo_level.test_used.predict.query import bigru_crf_jamo_pred
from _models.Bi_GRU_CRF.char_level.test_used.predict.query import bigru_crf_char_pred
from _models.ELMo_CRF.word_level.test_used.predict.query import elmo_crf_word_pred
from _models.ELMo_CRF.jamo_level.test_used.predict.query import elmo_crf_jamo_pred
from _models.ELMo_CRF.char_level.test_used.predict.query import elmo_crf_char_pred


absolute_path = "D:/LOTTE/" # 절대 경로 수정 필수
train_method = 'train_split' # test_used, train_split

# 아래 models 리스트의 model 중 추론에 사용할 모델 select
model = 'KoBERT_BiGRU_CRF'

models = ['BERT_CRF', 'DistilKoBERT_CRF', 'KoBERT', 'KoBERT_CRF', 'KoBERT_BiGRU_CRF', 'KoBERT_BiLSTM_CRF',
          'KoELECTRA', 'KoELECTRA_CRF', 'Bi_GRU_CRF-word-w2v_emb', 'Bi_GRU_CRF-word-fastText_emb', 'Bi_GRU_CRF-word-glove_emb',
          'Bi_GRU_CRF-word-elmo_emb', 'Bi_GRU_CRF-char-w2v_emb', 'Bi_GRU_CRF-char-fastText_emb', 'Bi_GRU_CRF-char-glove_emb',
          'Bi_GRU_CRF-char-elmo_emb', 'Bi_GRU_CRF-jamo-w2v_emb', 'Bi_GRU_CRF-jamo-fastText_emb', 'Bi_GRU_CRF-jamo-glove_emb',
          'Bi_GRU_CRF-jamo-elmo_emb', 'ELMo_CRF-word-w2v_emb', 'ELMo_CRF-word-fastText_emb', 'ELMo_CRF-word-glove_emb',
          'ELMo_CRF-word-elmo_emb', 'ELMo_CRF-char-w2v_emb', 'ELMo_CRF-char-fastText_emb', 'ELMo_CRF-char-glove_emb',
          'ELMo_CRF-char-elmo_emb', 'ELMo_CRF-jamo-w2v_emb', 'ELMo_CRF-jamo-fastText_emb', 'ELMo_CRF-jamo-glove_emb', 'ELMo_CRF-jamo-elmo_emb']

txt = "안녕하십니까, 한양대학교 데이터베이스 연구실을 졸업한 박승진입니다. 현재는 경희대학교 행정팀에 근무하고 있습니다."
# txt = '자기소개서.docx'
# txt = 'introduce_form1.xlsx'
# txt = 'introduce_form2.xlsx'
# txt = 'career_form1.xlsx'
# txt = 'career_form2.xlsx'
# txt = 'career_form3.xlsx'
# txt = 'pp_form1.xlsx'
# txt = 'pp_form2.xlsx'
# txt = 'pp_form3.xlsx'
# txt = 'security_form1.docx'
# txt = 'security_form2.docx'
# txt = 'security_form3.docx'


# 모델명 제대로 기입했는지 검사
if not any(model in m for m in models):
    print("Please fill in the model name correctly.")
    exit(1)

if model == 'BERT_CRF':
    result, rows, cols = bert_crf_pred(absolute_path=absolute_path,
                  model_name=model,
                  train_method=train_method,
                  txt=txt)
elif model == 'DistilKoBERT_CRF':
    result, rows, cols = distilkobert_crf_pred(absolute_path=absolute_path,
                          model_name=model,
                          train_method=train_method,
                          txt=txt)
elif model == 'KoBERT':
    result, rows, cols = kobert_pred(absolute_path=absolute_path,
                model_name=model,
                train_method=train_method,
                txt=txt)
elif model == 'KoBERT_CRF':
    result, rows, cols = kobert_crf_pred(absolute_path=absolute_path,
                    model_name=model,
                    train_method=train_method,
                    txt=txt)
elif model == 'KoBERT_BiGRU_CRF':
    result, rows, cols = kobert_bigru_crf_pred(absolute_path=absolute_path,
                          model_name=model,
                          train_method=train_method,
                          txt=txt)
elif model == 'KoBERT_BiLSTM_CRF':
    result, rows, cols = kobert_bilstm_crf_pred(absolute_path=absolute_path,
                           model_name=model,
                           train_method=train_method,
                           txt=txt)
elif model == 'KoELECTRA':
    result, rows, cols = koelectra_pred(absolute_path=absolute_path,
                           model_name=model,
                           train_method=train_method,
                           txt=txt)
elif model == 'KoELECTRA_CRF':
    result, rows, cols = koelectra_crf_pred(absolute_path=absolute_path,
                           model_name=model,
                           train_method=train_method,
                           txt=txt)
elif 'Bi_GRU_CRF' in model:
    tmp = model.split('-')
    model_name = tmp[0]
    level = tmp[1] + "_level"
    emb = tmp[2]

    if level == 'word_level':
        result, rows, cols = bigru_crf_word_pred(absolute_path=absolute_path,
                       model_name=model_name,
                       train_method=train_method,
                       level=level,
                       emb=emb,
                       txt=txt)
    if level == 'char_level':
        result, rows, cols = bigru_crf_char_pred(absolute_path=absolute_path,
                       model_name=model_name,
                       train_method=train_method,
                       level=level,
                       emb=emb,
                       txt=txt)
    if level == 'jamo_level':
        result, rows, cols = bigru_crf_jamo_pred(absolute_path=absolute_path,
                       model_name=model_name,
                       train_method=train_method,
                       level=level,
                       emb=emb,
                       txt=txt)
elif 'ELMo_CRF' in model:
    tmp = model.split('-')
    model_name = tmp[0]
    level = tmp[1] + "_level"
    emb = tmp[2]

    if level == 'word_level':
        result, rows, cols = elmo_crf_word_pred(absolute_path=absolute_path,
                       model_name=model_name,
                       train_method=train_method,
                       level=level,
                       emb=emb,
                       txt=txt)
    if level == 'char_level':
        result, rows, cols = elmo_crf_char_pred(absolute_path=absolute_path,
                       model_name=model_name,
                       train_method=train_method,
                       level=level,
                       emb=emb,
                       txt=txt)
    if level == 'jamo_level':
        result, rows, cols = elmo_crf_jamo_pred(absolute_path=absolute_path,
                       model_name=model_name,
                       train_method=train_method,
                       level=level,
                       emb=emb,
                       txt=txt)

print(result)
print(rows)
print(cols)
