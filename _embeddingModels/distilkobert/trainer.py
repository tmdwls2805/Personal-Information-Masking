import torch
from tqdm import trange
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
# from sklearn.metrics import classification_report
import logging
import pickle
from _utils.kobert.utils import classification_report_to_txt
from _utils.common.classification_report import classification_report
from _regex.regex import TokenMasker
from konlpy.tag import Okt
logger = logging.getLogger(__name__)

okt = Okt()

class Trainer:
    def __init__(self, config, train_dataset, val_dataset, model, label_list):
        self.config = config
        self.tr_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = model
        # GPU 사용가능할 경우 GPU 사용하고, GPU 사용 불가능시 CPU 사용
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_list = label_list

    def train(self):
        self.model.to(self.device)
        epoch = self.config.epochs
        t_total = len(self.tr_dataset) // self.config.gradient_accumulation_steps * epoch
        no_decay = ['bias', 'LayerNorm.weight']
        # Fine-tuning
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.01},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=self.config.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.config.num_warmup_steps
                                                    , num_training_steps=t_total)
        tag_set = ', '.join(self.label_list)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num training data = {} x {}".format(len(self.tr_dataset), self.config.batch_size))
        logger.info("  Num validation data = {} x {}".format(len(self.val_dataset), self.config.batch_size))
        logger.info("  Tag sets = [" + tag_set + "]")
        logger.info("  Num Epochs = %d", self.config.epochs)
        logger.info("  Total train batch size = %d", self.config.batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.config.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.model.zero_grad()
        for epochs in trange(epoch, desc="Epoch"):
            epochs = epochs + 1
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(self.tr_dataset):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                token_ids, attn_masks, labels = batch

                inputs = {'input_ids': token_ids,
                          'attn_masks': attn_masks,
                          'labels': labels}
                loss = self.model(**inputs)
                loss.backward()
                nb_tr_examples += token_ids.size(0)
                nb_tr_steps += 1
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()

                if step % self.config.batch_size == 0:
                    logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {}'.format(epochs, step * len(token_ids),
                                                                                     len(self.tr_dataset.dataset),
                                                                                     100. * step / len(self.tr_dataset),
                                                                                     loss.item()))
            eval_loss, eval_accuracy, predictions, true_labels = self.evaluate(val_dataset=self.val_dataset)
            logger.info('[{}] Eval Loss: {}, Eval Accuracy: {}%'.format(epochs, eval_loss, eval_accuracy * 100))
            if epochs == epoch:
                # classification report 출력시 report에 출력할 태그들 list로 만듦
                label_index_to_print = [i for i, label in enumerate(self.label_list) if label != "O"] # 'O' 태그 제외
                """ exobrain 데이터셋 같은 경우 커스텀 classification_report 사용불가함
                    exobrain 테스트 할때는 아래 classification_report 사용"""
                # report = classification_report(true_labels, predictions, target_names=self.target_names(), labels=label_index_to_print)
                """lotte, naver 데이터셋 학습시 아래 classification_report 사용하면 ~_B 태그와 ~_I 태그 둘다 맞았을 경우만 성능계산에 포함시킴"""
                report, accuracy = classification_report(true_labels, predictions, eval_accuracy)
                classification_report_to_txt(self.config, accuracy, './result/', report)
                print(report)
        self.save_model()

    def evaluate(self, val_dataset):
        self.model.eval()
        nb_eval_steps = 0
        eval_loss, eval_accuracy = 0, 0
        predictions, true_labels = [], []
        for batch in val_dataset:
            batch = tuple(t.to(self.device) for t in batch)
            token_ids, attn_masks, labels = batch
            inputs = {'input_ids': token_ids,
                      'attn_masks': attn_masks,
                      'labels': None
                      }
            val_inputs = {'input_ids': token_ids,
                          'attn_masks': attn_masks,
                          'labels': labels
                          }

            with torch.no_grad():
                tag_seqs = self.model(**inputs)
                tmp_eval_loss = self.model(**val_inputs)

            label_ids = labels.to('cpu').numpy()
            out_labels = label_ids.flatten()

            tmp_preds = []
            for seq in tag_seqs:
                tmp_preds.extend(seq)
            pred_labels = np.array(tmp_preds)

            # 커스텀 classification_report(B, I 둘다 맞아야 되는..) input 형태로 바꿔주기 위해
            # index output을 tag output(ex. PER_B)로 바꿔줌
            tmp_preds_tags = []
            for i in pred_labels:
                pred = self.label_list[i]
                tmp_preds_tags.append(pred)

            tmp_true_labels = []
            for i in out_labels:
                true = self.label_list[i]
                tmp_true_labels.append(true)

            predictions.extend(tmp_preds_tags)
            true_labels.extend(tmp_true_labels)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += self.flat_accuracy(pred_labels, out_labels)
            nb_eval_steps += 1

        loss = eval_loss / nb_eval_steps
        accuracy = eval_accuracy / nb_eval_steps
        return loss, accuracy, predictions, true_labels

    def flat_accuracy(self, preds, labels):
        return np.sum(preds == labels) / len(labels)

    # 훈련된 model 저장하는 함수
    def save_model(self):
        torch.save(self.model, './model/distilkobert_{}_{}.pth'.format(self.config.epochs, self.config.data_name))

    # 'O' 태그 제외하고 성능 계산하기 위한 용도로, classification report 인자로 쓰임
    def target_names(self):
        target_names = []
        for ner_tag in self.label_list:
            if ner_tag == "O":
                continue
            else:
                target_names.append(ner_tag)
        print("target names : ", target_names)
        return target_names


def predict(padded_raw_data, model, unique_labels, out_file_name):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token_ids, attn_mask, org_tok_map, sorted_idx, original_tokens = padded_raw_data
    w = open(out_file_name, 'w', encoding='utf-8')

    inputs = {'input_ids': token_ids.to(device),
              'attn_masks': attn_mask.to(device),
              'labels': None
              }
    with torch.no_grad():
        tag_seqs = model(**inputs)

    txt = ""

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

    filter = ['JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX',
              'JC', 'VCP', 'EC', 'MAJ', 'VCP+EF', 'SF', 'XSN', 'SC',
              'VCP+ETM', 'NNB', 'VCP+EC', 'EP+EC', 'ETM']

    prev_token_pos = ''

    # encoding된 input original token들을 다시 문자로 decoding
    for i in range(len(sorted_idx)):
        o2m = org_tok_map[i]
        pos = sorted_idx.index(i)
        for j, orig_tok_idx in enumerate(o2m):
            unique_label = unique_labels[tag_seqs[pos][orig_tok_idx]]
            original_token = original_tokens[i][j]
            w.write(original_token + '\t')
            w.write(unique_label + '\n')
            # 마스킹하고자 하는 토큰
            o_token = ''
            # 조사, 형용사 등 나머지 토큰
            rem_token = ''
            # print(okt.pos(original_token))
            if unique_label is not 'O':
                r = okt.pos(original_token)
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
                txt += '*' * len(o_token) + rem_token + ' '
                if 'PER' in unique_label:
                    if unique_label == 'PER_B' and per_count > 0:
                        per_count += 1
                        rows_per_txt += ', ' + o_token
                    elif unique_label == 'PER_B':
                        per_count += 1
                        rows_per_txt += o_token
                    else:
                        rows_per_txt += ' ' + o_token
                elif 'LOC' in unique_label:
                    if unique_label == 'LOC_B' and loc_count > 0:
                        loc_count += 1
                        rows_loc_txt += ', ' + o_token
                    elif unique_label == 'LOC_B':
                        loc_count += 1
                        rows_loc_txt += o_token
                    else:
                        rows_loc_txt += ' ' + o_token
                elif 'AFF' in unique_label:
                    if unique_label == 'AFF_B' and aff_count > 0:
                        aff_count += 1
                        rows_aff_txt += ', ' + o_token
                    elif unique_label == 'AFF_B':
                        aff_count += 1
                        rows_aff_txt += o_token
                    else:
                        rows_aff_txt += ' ' + o_token
                elif 'EDU' in unique_label:
                    if unique_label == 'EDU_B' and edu_count > 0:
                        edu_count += 1
                        rows_edu_txt += ', ' + o_token
                    elif unique_label == 'EDU_B':
                        edu_count += 1
                        rows_edu_txt += o_token
                    else:
                        rows_edu_txt += ' ' + o_token
                elif 'POS' in unique_label:
                    if unique_label == 'POS_B' and pos_count > 0:
                        pos_count += 1
                        rows_pos_txt += ', ' + o_token
                    elif unique_label == 'POS_B':
                        pos_count += 1
                        rows_pos_txt += o_token
                    else:
                        rows_pos_txt += ' ' + o_token
                elif 'COM' in unique_label:
                    if unique_label == 'COM_B' and com_count > 0:
                        com_count += 1
                        rows_com_txt += ', ' + o_token
                    elif unique_label == 'COM_B':
                        com_count += 1
                        rows_com_txt += o_token
                    else:
                        rows_com_txt += ' ' + o_token
            else:
                result, tag = TokenMasker(original_token).examine()
                if result is not None:
                    txt += result[0] + ' '
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
                else:
                    txt += original_token + ' '
        w.write('\n')
    w.close()

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

    logger.info("Raw data prediction done!")
    logger.info("Result saved as csv file")

    return txt, rows_out, cols_out

def load_model(path, config, absolute_path, model_name, train_method):
    f = open(absolute_path + '_models/' + model_name + '/' + train_method + '/' + 'ner_vocab_{}.pkl'.format(config.data_name), 'rb')
    tag2idx = pickle.load(f)
    unique_labels = list(tag2idx.keys())

    return torch.load(path), unique_labels, tag2idx
