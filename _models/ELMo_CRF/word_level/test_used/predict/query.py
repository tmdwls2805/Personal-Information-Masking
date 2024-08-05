from keras.models import model_from_json
from _utils.common.Mecab import Mecab
from _utils.common.config import Config
import numpy as np
from keras_contrib.layers import CRF
from _regex.regex import TokenMasker
from _parser.parser_sentence import pred_data_parsing
from _utils.common.Mecab import Mecab
import kss

me = Mecab()

def elmo_crf_word_pred(absolute_path, train_method, model_name, level, emb, txt):

    config = Config(json_path=absolute_path + "_models/" + model_name + "/" + level + "/" + train_method + "/config.json")

    load_data = config.training_data
    epochs = config.epochs
    absolute_path = config.absolute_path

    model_save_file_path = absolute_path + '_save/Elmo-CRF/word_level/' + train_method + '/model/'+ emb +'/'
    data_save_file_path = absolute_path + '_save/Elmo-CRF/word_level/' + train_method + '/data/'+ emb +'/'

    model = model_save_file_path + load_data + "/" + load_data + '_' + str(epochs) + "_" + emb + ".json"
    model_weight = model_save_file_path + load_data + "/" + load_data + '_' + str(epochs) + "_" + emb + ".h5"

    word_to_index = data_save_file_path + load_data + "/" + load_data + '_' + str(epochs) + "_word_to_index.npy"
    pos_to_index = data_save_file_path + load_data + "/" + load_data + '_' + str(epochs) + "_pos_to_index.npy"
    ner_to_index = data_save_file_path + load_data + "/" + load_data + '_' + str(epochs) + "_ner_to_index.npy"
    index_to_word = data_save_file_path + load_data + "/" + load_data + '_' + str(epochs) + "_index_to_word.npy"
    index_to_ner = data_save_file_path + load_data + "/" + load_data + '_' + str(epochs) + "_index_to_ner.npy"

    json_file = open(model, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json, custom_objects={"CRF": CRF})
    loaded_model.load_weights(model_weight)

    loaded_word_to_index = np.load(word_to_index, allow_pickle=True)
    loaded_ner_to_index = np.load(ner_to_index, allow_pickle=True)
    loaded_pos_to_index = np.load(pos_to_index, allow_pickle=True)
    loaded_index_to_word = np.load(index_to_word, allow_pickle=True)
    loaded_index_to_ner = np.load(index_to_ner, allow_pickle=True)

    me = Mecab()

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

    masked_sents = []
    per_count, loc_count, aff_count, edu_count, pos_count = 0, 0, 0, 0, 0
    com_count, rrn_count, tel_count, date_count, mail_count = 0, 0, 0, 0, 0

    rows_per = ['PER']
    rows_loc = ['LOC']
    rows_aff = ['AFF']
    rows_edu = ['EDU']
    rows_pos = ['POS']
    rows_com = ['COM']
    rows_rrn = ['RRN']
    rows_tel = ['TEL']
    rows_date = ['DATE']
    rows_mail = ['MAIL']

    cols_per = ['PER']
    cols_loc = ['LOC']
    cols_aff = ['AFF']
    cols_edu = ['EDU']
    cols_pos = ['POS']
    cols_com = ['COM']
    cols_rrn = ['RRN']
    cols_tel = ['TEL']
    cols_date = ['DATE']
    cols_mail = ['MAIL']

    filter = ['JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX',
              'JC', 'VCP', 'EC', 'MAJ', 'VCP+EF', 'SF', 'XSN', 'SC',
              'VCP+ETM', 'NNB', 'VCP+EC', 'EP+EC', 'ETM']

    rows_per_txt = ''
    rows_loc_txt = ''
    rows_aff_txt = ''
    rows_edu_txt = ''
    rows_pos_txt = ''
    rows_com_txt = ''
    rows_rrn_txt = ''
    rows_tel_txt = ''
    rows_date_txt = ''
    rows_mail_txt = ''

    prev_token_pos = ''

    for new_sentence in sentences:
        split_sentence = new_sentence.split()

        sentence = []
        pos = []
        f_input = []

        for w in new_sentence.split(' '):
            f_input.extend(me.pos(w))
        for word in f_input:
            sentence.append(word[0])
            pos.append(word[1])

        new_X = []
        for w in sentence:
            new_X.append(loaded_word_to_index.item().get(w, 1))

        new_y = []
        for w in pos:
            new_y.append(loaded_pos_to_index.item().get(w, 1))

        from keras.preprocessing import sequence
        max_len = config.maxlen
        test_sent = sequence.pad_sequences([new_X], maxlen=max_len, truncating='post', padding='post')
        test_pos = sequence.pad_sequences([new_y], maxlen=max_len, truncating='post', padding='post')

        this_pred = loaded_model.predict([test_sent, test_pos])
        y_predicted = np.argmax(this_pred, axis=-1) # 원-핫 인코딩을 다시 정수 인코딩으로 변경함.

        out_file_name = absolute_path + "_models/" + model_name + "/" + level + "/" + train_method + \
                        "/result/raw_prediction_{}_{}_{}.csv".format(emb, epochs, config.data_name)
        writer = open(out_file_name, 'a', encoding='utf-8')

        # print("{:15}|{:5}|{}".format("입력단어", "훈련단어", "예측값"))
        # print(35 * "-")

        for w, t, pred in zip(sentence, test_sent[0], y_predicted[0]):
            if t != 0:  # PAD값은 제외함.
                writer.write(w + '\t')
                writer.write(loaded_index_to_ner.item()[pred] + '\n')
                # print("{:17}: {:7} {}".format(w, loaded_index_to_word.item()[t], loaded_index_to_ner.item()[pred]))
        writer.write("\n")
        writer.close()

        tmp_token_list = []
        tmp_tag_list = []
        tmp_tag_list2 = []
        start = 0
        end = 0

        for t, pred in zip(test_sent[0], y_predicted[0]):
            if t != 0:
                tmp_tag_list.append(loaded_index_to_ner.item()[pred])

        for token, t, pred in zip(split_sentence, test_sent[0], y_predicted[0]):
            h = me.pos(token)
            end = len(h) + start
            tmp_token_list.append(sentence[start:end])
            tmp_tag_list2.append(tmp_tag_list[start:end])
            start = end

        s = ''
        for token, tag in zip(tmp_token_list, tmp_tag_list2):
            # 마스킹하고자 하는 토큰
            o_token = ''
            # 조사, 형용사 등 나머지 토큰
            rem_token = ''
            tmp_token = ''.join(token)
            if tag[0] == 'O':
                s += tmp_token + ' '
            else:
                r = me.pos(tmp_token)
                for a in r:
                    if a[1] == 'NNG' and a[0] == '의':
                        rem_token += a[0]
                        prev_token_pos = a[1]
                    elif a[1] == 'XSN' and a[0] == '화' and prev_token_pos == 'NNG':
                        o_token += a[0]
                    elif a[1] not in filter:
                        o_token += a[0]
                        prev_token_pos = a[1]
                    else:
                        rem_token += a[0]
                s += '*' * len(o_token) + rem_token + ' '
                if 'PER' in tag[0]:
                    if tag[0] == 'PER_B' and per_count > 0:
                        per_count += 1
                        rows_per_txt += ', ' + o_token
                    elif tag[0] == 'PER_B':
                        per_count += 1
                        rows_per_txt += o_token
                    else:
                        rows_per_txt += ' ' + o_token
                elif 'LOC' in tag[0]:
                    if tag[0] == 'LOC_B' and loc_count > 0:
                        loc_count += 1
                        rows_loc_txt += ', ' + o_token
                    elif tag[0] == 'LOC_B':
                        loc_count += 1
                        rows_loc_txt += o_token
                    else:
                        rows_loc_txt += ' ' + o_token
                elif 'AFF' in tag[0]:
                    if tag[0] == 'AFF_B' and aff_count > 0:
                        aff_count += 1
                        rows_aff_txt += ', ' + o_token
                    elif tag[0] == 'AFF_B':
                        aff_count += 1
                        rows_aff_txt += o_token
                    else:
                        rows_aff_txt += ' ' + o_token
                elif 'EDU' in tag[0]:
                    if tag[0] == 'EDU_B' and edu_count > 0:
                        edu_count += 1
                        rows_edu_txt += ', ' + o_token
                    elif tag[0] == 'EDU_B':
                        edu_count += 1
                        rows_edu_txt += o_token
                    else:
                        rows_edu_txt += ' ' + o_token
                elif 'POS' in tag[0]:
                    if tag[0] == 'POS_B' and pos_count > 0:
                        pos_count += 1
                        rows_pos_txt += ', ' + o_token
                    elif tag[0] == 'POS_B':
                        pos_count += 1
                        rows_pos_txt += o_token
                    else:
                        rows_pos_txt += ' ' + o_token
                elif 'COM' in tag[0]:
                    if tag[0] == 'COM_B' and com_count > 0:
                        com_count += 1
                        rows_com_txt += ', ' + o_token
                    elif tag[0] == 'COM_B':
                        com_count += 1
                        rows_com_txt += o_token
                    else:
                        rows_com_txt += ' ' + o_token
        last_output = ''
        for t in s.split(' '):
            result, tag = TokenMasker(t).examine()
            if result is None:
                last_output += t + ' '
            else:
                last_output += result[0] + ' '
                if tag == 'RRN':
                    rrn_count += 1
                    rows_rrn_txt += result[1] + ', '
                elif tag == 'TEL':
                    tel_count += 1
                    rows_tel_txt += result[1] + ', '
                elif tag == 'DATE':
                    date_count += 1
                    rows_date_txt += result[1] + ', '
                elif tag == 'MAIL':
                    mail_count += 1
                    rows_mail_txt += result[1] + ', '
        masked_sents.append(last_output)
    result = ''.join(masked_sents)

    if len(rows_per_txt) > 0 and rows_per_txt[-1] == ' ':
        rows_per_txt = rows_per_txt[:-2]
    if len(rows_loc_txt) > 0 and rows_loc_txt[-1] == ' ':
        rows_loc_txt = rows_loc_txt[:-2]
    if len(rows_aff_txt) > 0 and rows_aff_txt[-1] == ' ':
        rows_aff_txt = rows_aff_txt[:-2]
    if len(rows_edu_txt) > 0 and rows_edu_txt[-1] == ' ':
        rows_edu_txt = rows_edu_txt[:-2]
    if len(rows_pos_txt) > 0 and rows_pos_txt[-1] == ' ':
        rows_pos_txt = rows_pos_txt[:-2]
    if len(rows_com_txt) > 0 and rows_com_txt[-1] == ' ':
        rows_com_txt = rows_com_txt[:-2]
    rows_rrn_txt = rows_rrn_txt[:-2]
    rows_tel_txt = rows_tel_txt[:-2]
    rows_date_txt = rows_date_txt[:-2]
    rows_mail_txt = rows_mail_txt[:-2]

    rows_per.append(rows_per_txt), rows_loc.append(rows_loc_txt), rows_aff.append(rows_aff_txt),
    rows_edu.append(rows_edu_txt), rows_pos.append(rows_pos_txt), rows_com.append(rows_com_txt),
    rows_rrn.append(rows_rrn_txt), rows_tel.append(rows_tel_txt), rows_date.append(rows_date_txt), rows_mail.append(rows_mail_txt)

    rows = []
    if '' not in rows_per:
        rows.append(rows_per)
    if '' not in rows_loc:
        rows.append(rows_loc)
    if '' not in rows_aff:
        rows.append(rows_aff)
    if '' not in rows_edu:
        rows.append(rows_edu)
    if '' not in rows_pos:
        rows.append(rows_pos)
    if '' not in rows_com:
        rows.append(rows_com)
    if '' not in rows_rrn:
        rows.append(rows_rrn)
    if '' not in rows_tel:
        rows.append(rows_tel)
    if '' not in rows_date:
        rows.append(rows_date)
    if '' not in rows_mail:
        rows.append(rows_mail)

    cols = []
    cols_per.append(per_count), cols_loc.append(loc_count), cols_aff.append(aff_count), cols_edu.append(edu_count),
    cols_pos.append(pos_count), cols_com.append(com_count), cols_rrn.append(rrn_count), cols_tel.append(tel_count),
    cols_date.append(date_count), cols_mail.append(mail_count)

    cols.append(cols_per), cols.append(cols_loc), cols.append(cols_aff), cols.append(cols_edu), cols.append(cols_pos),
    cols.append(cols_com), cols.append(cols_rrn), cols.append(cols_tel), cols.append(cols_date), cols.append(cols_mail)

    rows_out = []
    for row in rows:
        if len(row) > 1:
            rows_out.append(row)

    cols_out = []
    for col in cols:
        if col[1] != 0:
            cols_out.append(col)

    return result, rows_out, cols_out