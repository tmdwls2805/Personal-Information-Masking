import pandas as pd
from _utils.common.Mecab import Mecab
import os

def career2_tag(x):

    data = pd.read_excel(x, header=None)
    data = data.fillna(0)

    first_data = []
    first_process1_data = []
    first_process2_data = []
    first_process3_data = []
    final_result = []

    columns1 = range(1, 31)
    rows = range(0, 10)

    for j in columns1:
        for i in rows:
            if data[i][j] != 0:
                first_data.append(data[i][j])

    for i in range(0, 7):
        if len(first_process1_data) % 2 == 0:
            first_process1_data.append(first_data[i])
        else:
            first_process1_data.append(first_data[i].replace(" ", ''))

    for j in range(7, 24):
        if len(first_process2_data) % 2 == 0:
            first_process2_data.append(first_data[j])
        else:
            first_process2_data.append(first_data[j].replace(" ", ''))

    for k in range(24, 25):
        first_process3_data.append(first_data[k])

    final_result.extend(first_process1_data)
    final_result.extend(first_process2_data)
    final_result.extend(first_process3_data)

    # Mecab

    me = Mecab()

    word_pos = []
    for i in range(1):
        word_list = []
        pos_list = []
        for word in final_result:
            word_set = me.pos(word)
            for w in word_set:
                word_list.append(w[0])
                pos_list.append(w[1])
        b_idx = 0
        e_idx = 0

        for i in range(len(word_list)):
            if word_list[i] == '성명':
                b_idx = i + 1
            if word_list[i] == '생년월일':
                e_idx = i

        temp = word_list[b_idx:e_idx]
        name = "".join(temp)

        del (word_list[b_idx:e_idx])
        del (pos_list[b_idx:e_idx])

        word_list.insert(b_idx, name)
        pos_list.insert(b_idx, "NNP")

        # print(word_list)
        # print(pos_list)

        for i in range(len(word_list)):
            word_pos.append(word_list[i] + " " + pos_list[i])

    all_data = list(filter(None, word_pos))


    all_data_tag = []
    dup_tag = []
    dup_tag2 = []
    dup_tag3 = []
    s_idx = 0
    se_idx = 0

    for i in range(len(all_data)):
        all_data_tag.append(all_data[i] + " " + "O")

    for n in range(len(all_data_tag)):
        if all_data_tag[n] == "경력 NNG O":
            dup_tag.append(n)

        if all_data_tag[n] == "담당 NNG O":
            dup_tag2.append(n)

        if all_data_tag[n] == "( SSO O":
            dup_tag3.append(n)

    for i in range(len(all_data_tag)):

        text = all_data_tag[i]

        # 인적사항 - 이름
        if all_data_tag[i] == "성명 NNG O" and s_idx == 0:
            s_idx = i + 1
        if all_data_tag[i] == "생년월일 NNG O" and se_idx == 0:
            se_idx = i
            tag = []
            for j in range(s_idx, se_idx):
                tag.append(all_data[j])
                for k in range(0, len(tag)):
                    if k == 0:
                        all_data_tag[j] = (tag[k] + " " + "PER_B")
                    else:
                        all_data_tag[j] = (tag[k] + " " + "PER_I")
            s_idx = 0
            se_idx = 0

        # 인적사항 - 이름
        if all_data_tag[i] == "주소 NNG O" and s_idx == 0:
            s_idx = i + 1
            se_idx = dup_tag[0]
            tag = []
            for j in range(s_idx, se_idx):
                tag.append(all_data[j])
                for k in range(0, len(tag)):
                    if k == 0:
                        all_data_tag[j] = (tag[k] + " " + "LOC_B")
                    else:
                        all_data_tag[j] = (tag[k] + " " + "LOC_I")
            s_idx = 0
            se_idx = 0

        # 경력사항 - 회사명
        if all_data_tag[i] == "명 NNG O" and s_idx == 0:
            s_idx = i + 1
        if all_data_tag[i] == "사업자 NNG O" and se_idx == 0:
            se_idx = i
            tag = []
            for j in range(s_idx, se_idx):
                tag.append(all_data[j])
                for k in range(0, len(tag)):
                    if k == 0:
                        all_data_tag[j] = (tag[k] + " " + "COM_B")
                    else:
                        all_data_tag[j] = (tag[k] + " " + "COM_I")
            s_idx = 0
            se_idx = 0

        # 경력사항 - 주소
        if all_data_tag[i] == "소재지 NNG O" and s_idx == 0:
            s_idx = i + 1
        if all_data_tag[i] == "근무 NNG O" and se_idx == 0:
            se_idx = i
            tag = []
            for j in range(s_idx, se_idx):
                tag.append(all_data[j])
                for k in range(0, len(tag)):
                    if k == 0:
                        all_data_tag[j] = (tag[k] + " " + "LOC_B")
                    else:
                        all_data_tag[j] = (tag[k] + " " + "LOC_I")
            s_idx = 0
            se_idx = 0

        # 경력사항 - 근무부서
        if all_data_tag[i] == "부서 NNG O" and s_idx == 0:
            s_idx = i + 1
        if all_data_tag[i] == "직위 NNG O" and se_idx == 0:
            se_idx = i
            tag = []
            for j in range(s_idx, se_idx):
                tag.append(all_data[j])
                for k in range(0, len(tag)):
                    if k == 0:
                        all_data_tag[j] = (tag[k] + " " + "AFF_B")
                    else:
                        all_data_tag[j] = (tag[k] + " " + "AFF_I")
            s_idx = 0
            se_idx = 0

        # 경력사항 - 직위
        if all_data_tag[i] == "직위 NNG O" and s_idx == 0 and se_idx == 0:
            s_idx = i + 1
            se_idx = dup_tag2[len(dup_tag2)-1]
            tag = []
            for j in range(s_idx, se_idx):
                tag.append(all_data[j])
                for k in range(0, len(tag)):
                    if k == 0:
                        all_data_tag[j] = (tag[k] + " " + "POS_B")
                    else:
                        all_data_tag[j] = (tag[k] + " " + "POS_I")
            s_idx = 0
            se_idx = 0

        # 비고 - 회사명
        if all_data_tag[i] == "일 NNBC O" and s_idx == 0 and se_idx == 0:
            s_idx = i + 1
            se_idx = dup_tag3[len(dup_tag3)-1]
            tag = []
            for j in range(s_idx, se_idx):
                tag.append(all_data[j])
                for k in range(0, len(tag)):
                    if k == 0:
                        all_data_tag[j] = (tag[k] + " " + "COM_B")
                    else:
                        all_data_tag[j] = (tag[k] + " " + "COM_I")
            s_idx = 0
            se_idx = 0

    return all_data_tag