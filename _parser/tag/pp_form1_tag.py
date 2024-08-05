import pandas as pd
from _utils.common.Mecab import Mecab

def pp1_tag(x):

    data = pd.read_excel(x, header=None)
    data = data.fillna(0)

    final_result = []
    first_data = []
    second_data = []

    columns1 = range(8, 18)
    rows = range(0, 10)

    for j in columns1:
        for i in rows:
            if data[i][j] != 0:
                first_data.append(str(data[i][j]))

    for i in range(22, 40, 2):
        if data[0][i] != 0:
            second_data.append(data[0][20])
            second_data.append(data[0][i])
        if data[2][i] != 0:
            second_data.append(data[2][20])
            second_data.append(data[2][i])
        if data[4][i] != 0:
            second_data.append(data[4][20])
            second_data.append(data[4][i])
        if data[7][i] != 0:
            second_data.append(data[7][20])
            second_data.append(data[7][i])
        if data[8][i] != 0:
            second_data.append(data[8][20])
            second_data.append(data[8][i])

    final_result.extend(first_data)
    final_result.extend(second_data)

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
            if word_list[i] == '소속':
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

    s_idx = 0
    se_idx = 0
    dup_tag = []
    for i in range(len(all_data)):
        all_data_tag.append(all_data[i] + " " + "O")

    for n in range(len(all_data_tag)):
        if all_data_tag[n] == ") SSC O":
            dup_tag.append(n)

    for i in range(len(all_data_tag)):

        # 성명
        if all_data_tag[i] == "성명 NNG O" and s_idx == 0:
            s_idx = i + 1
        if all_data_tag[i] == "소속 NNG O" and se_idx == 0:
            se_idx = i
            tag = []
            for j in range(s_idx, se_idx):
                tag.append(all_data[j])
                for k in range(0, len(tag)):
                    if k == 0:
                        all_data_tag[j] = (tag[k] + " " + "PER_B")
                    else:
                        all_data_tag[j] = (tag[k] + " " + "PER_I")
            se_idx = 0
            s_idx = 0

        # 소속
        if all_data_tag[i] == "소속 NNG O" and s_idx == 0:
            s_idx = i + 1
        if all_data_tag[i] == "최종 NNG O" and se_idx == 0:
            se_idx = i
            tag = []
            for j in range(s_idx, se_idx):
                tag.append(all_data[j])
                for k in range(0, len(tag)):
                    if k == 0:
                        all_data_tag[j] = (tag[k] + " " + "AFF_B")
                    else:
                        all_data_tag[j] = (tag[k] + " " + "AFF_I")
            se_idx = 0
            s_idx = 0

            # 최종학력(학위)
        if all_data_tag[i] == "전공 NNG O" and se_idx == 0 and s_idx == 0:
            se_idx = i
            s_idx = dup_tag[0] + 1
            tag = []
            for j in range(s_idx, se_idx):
                tag.append(all_data[j])
                for k in range(0, len(tag)):
                    if k == 0:
                        all_data_tag[j] = (tag[k] + " " + "EDU_B")
                    else:
                        all_data_tag[j] = (tag[k] + " " + "EDU_I")

    return all_data_tag