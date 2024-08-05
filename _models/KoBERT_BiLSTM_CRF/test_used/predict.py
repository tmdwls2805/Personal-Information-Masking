from kobert_transformers import get_tokenizer
from _utils.common.config import Config
from _embeddingModels.kobert.data_processing import raw_processing
from _embeddingModels.kobert.trainer import load_model, predict
from _utils.kobert.utils import init_logger, set_seed
from _parser.parser_sentence import pred_data_parsing
import kss


def kobert_bilstm_crf_pred(absolute_path, model_name, train_method, txt):
    # *** Prediction ***
    set_seed()
    init_logger()
    config = Config(json_path=absolute_path + "_models/" + model_name + "/" + train_method + "/" + "./config.json")

    if '.docx' in txt:
        sentences = sum(pred_data_parsing(absolute_path, txt), [])
    elif '.xlsx' in txt:
        sentences = sum(pred_data_parsing(absolute_path, txt), [])
    else:
        sentences = kss.split_sentences(txt)

    with open(absolute_path + '_models/' + model_name + '/' + train_method +
              '/result/parsed_result.txt', 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')
        f.close()

    # tokenizer load
    tokenizer = get_tokenizer()
    # 저장된 모델 및 정답 태그 load
    model, unique_labels, _ = load_model(absolute_path + "_models/" + model_name + "/" + train_method +
                                         './model/kobert-{}_{}_{}.pth'.format(config.model_name, config.epochs, config.data_name),
                                         config=config,
                                         absolute_path=absolute_path,
                                         model_name=model_name,
                                         train_method=train_method)
    # input raw data padding
    pad_data = raw_processing(sentences, tokenizer)
    # out_file_name --> 예측 결과 문서파일 저장되는 경로
    result, rows, cols = predict(pad_data, model, unique_labels,
                                 out_file_name=absolute_path + '_models/' + model_name + '/' + train_method +
                                            './result/raw_prediction_{}_{}.csv'.format(config.epochs, config.data_name))

    return result, rows, cols