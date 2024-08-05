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

txt = "이름은 서동국이고 생년월일은 93년 04월 06일 입니다. " \
      "그리고 저는 현재 경기도 용인시 수지구에 거주하고 있으며, 어렸을 때에는 20개월동안 미국에서 살다왔습니다. " \
      "제 주민등록번호는 930406-1243435이며 한양대학교 데이터베이스 연구실을 졸업했습니다. "\
      "제가 다니고 싶은 회사는 (주)삼일제지이고 거기서 NLP 고도화팀에 들어가서 상무 이상의 직급이 되고 싶습니다. "\
      "더 궁금하신 사항 있으시면 _sdk6789_@hanyang.ac.kr로 메일주시거나 010-1234-5678로 문자주시기 바랍니다. 감사합니다."
txt = "저는 작년에 종근당 건강과 티맥스 소프트, IBK 신용정보원에 지원하였었습니다."
# txt = "인재경영부 전지원 주임 입니다. 요청하신 경력증명서를 보내 드립니다."
# txt = "안녕하십니까, 롯데정보통신에 다니는 이웅기입니다. " \
#       "3월 5일에 입사하였고 현재 나이는 스물아홉 입니다. " \
#       "저의 메일 주소는 OK_flag@naver.com이고 연락처는 010-7487-0125 입니다. " \
#       "현재는 한양대학교와 산학협력을 진행하고 있습니다. 감사합니다."
# txt = "군 제대 후의 롯데마트에서 FO로써 고객관리, 직원관리 등의 소중함을 느꼈고, 인터넷 쇼핑몰에서의 주임 활동은 인터넷쇼핑몰의 고객들의 불만 처리, 제품의 매출인상을 위한 마케팅등을 배웠습니다."
# txt = "고정관념에 얽메이지 않는 창의적인 사고로 고객과 가맹점주, GS25 모두에게 이익과 감동을 줄 Win-Win 전략을 만들어야 합니다."
# txt = "고객을 최우선으로 하여 고객의 자산을 제 것 처럼 소중하게 다루어 고객에게 만족과 신뢰를 드릴 준비가 되어있는 예비 대우증권인 김성도 입니다."
# txt = "안녕하십니까, 한양대학교 데이터베이스 연구실을 졸업한 박승진입니다. 현재는 경희대학교 행정팀에 근무하고 있습니다."
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