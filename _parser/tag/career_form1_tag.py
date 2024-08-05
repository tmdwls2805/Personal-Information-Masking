import pandas as pd
from _utils.common.Mecab import Mecab
from _parser.utils import replaceRight

def career1_tag(x):

    data = pd.read_excel(x, header=None)
    data = data.fillna(0)

    final_result = []
    first_data = []
    second_data = []
    third_data = []
    all_data = []

    columns1 = range(8, 22)
    columns2 = range(32, 37)
    rows = range(0, 10)

    for j in columns1:
        for i in rows:
            if data[i][j] != 0:
                first_data.append(data[i][j])
                # print(data[i][j], end=' ')

    second_data.append(data[0][22])
    for i in range(24, 31, 2):
        if data[1][i] != 0:
            second_data.append(data[1][22])
            second_data.append(data[1][i])
        if data[5][i] != 0:
            second_data.append(data[5][22])
            second_data.append(data[5][i])
        if data[7][i] != 0:
            second_data.append(data[7][22])
            second_data.append(data[7][i])

    for j in columns2:
        for i in rows:
            if data[i][j] != 0:
                third_data.append(data[i][j])

                # print(data[i][j], end=' ')
    if len(second_data) == 0:
        final_result.extend(first_data)
        final_result.extend(third_data)

    elif len(second_data) != 0:
        final_result.extend(first_data)
        final_result.extend(second_data)
        final_result.extend(third_data)

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
            if word_list[i] == '용도':
                e_idx = i

        for i in range(len(word_list)):
            if word_list[i] == '이름':
                b_idx = i + 1
            if word_list[i] == '학력':
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
    dup_tag4 = []
    dup_tag5 = []
    dup_tag6 = []
    dup_tag7 = []

    s_idx = 0
    se_idx = 0
    for i in range(len(all_data)):
        all_data_tag.append(all_data[i] + " " + "O")

    for n in range(len(all_data_tag)):
        if all_data_tag[n] == "주소 NNG O":
            dup_tag.append(n)

        if all_data_tag[n] == "업체 NNG O":
            dup_tag2.append(n)

        if all_data_tag[n] == "경력 NNG O":
            dup_tag3.append(n)

        if all_data_tag[n] == "대표 NNG O":
            dup_tag4.append(n)

        if all_data_tag[n] == "부서 NNG O":
            dup_tag5.append(n)

        if all_data_tag[n] == "담당 NNG O":
            dup_tag6.append(n)

        if all_data_tag[n] == "( SSO O":
            dup_tag7.append(n)

    rp = len(dup_tag4) - 1
    bk = len(dup_tag7) - 1

    for i in range(len(all_data_tag)):

        # 인적사항 - 이름
        if all_data_tag[i] == "이름 NNG O" and s_idx == 0:
            s_idx = i + 1
        if all_data_tag[i] == "학력 NNG O" and se_idx == 0:
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

        # 인적사항 - 학력
        if all_data_tag[i] == "학력 NNG O" and s_idx == 0:
            s_idx = i + 1
        if all_data_tag[i] == "소속 NNG O" and se_idx == 0:
            se_idx = i
            tag = []
            for j in range(s_idx, se_idx):
                tag.append(all_data[j])
                for k in range(0, len(tag)):
                    if k == 0:
                        all_data_tag[j] = (tag[k] + " " + "EDU_B")
                    else:
                        all_data_tag[j] = (tag[k] + " " + "EDU_I")
            se_idx = 0
            s_idx = 0

        # 인적사항 - 소속
        if all_data_tag[i] == "소속 NNG O" and s_idx == 0:
            s_idx = i + 1
        if all_data_tag[i] == "주민 NNG O" and se_idx == 0:
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

        # 인적사항 - 주소
        if all_data_tag[i] == "업체 NNG O" and s_idx == 0 and se_idx == 0:
            s_idx = dup_tag[0] + 1
            se_idx = dup_tag2[0]
            tag = []
            for j in range(s_idx, se_idx):
                tag.append(all_data[j])
                for k in range(0, len(tag)):
                    if k == 0:
                        all_data_tag[j] = (tag[k] + " " + "LOC_B")
                    else:
                        all_data_tag[j] = (tag[k] + " " + "LOC_I")
            se_idx = 0
            s_idx = 0

        # 업체정보 - 업체명(회사명)
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
            se_idx = 0
            s_idx = 0

        # 업체정보 - 주소
        if all_data_tag[i] == "경력 NNG O" and s_idx == 0 and se_idx == 0:
            s_idx = dup_tag[1] + 1
            se_idx = dup_tag3[0]
            tag = []
            for j in range(s_idx, se_idx):
                tag.append(all_data[j])
                for k in range(0, len(tag)):
                    if k == 0:
                        all_data_tag[j] = (tag[k] + " " + "LOC_B")
                    else:
                        all_data_tag[j] = (tag[k] + " " + "LOC_I")
            se_idx = 0
            s_idx = 0

        # 경력사항 - 근무부서(부서1, 부서2)
        if all_data_tag[i] == "부서 NNG O" and s_idx == 0:
            for m in range(0, len(dup_tag5)):
                s_idx = dup_tag5[m - 1] + 1
                se_idx = dup_tag6[m - 1]
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

        # 확인자 - 직위
        if all_data_tag[i] == "직위 NNG O" and s_idx == 0:
            s_idx = i + 1
        if all_data_tag[i] == "성명 NNG O" and se_idx == 0:
            se_idx = i
            tag = []
            for j in range(s_idx, se_idx):
                tag.append(all_data[j])
                for k in range(0, len(tag)):
                    if k == 0:
                        all_data_tag[j] = (tag[k] + " " + "POS_B")
                    else:
                        all_data_tag[j] = (tag[k] + " " + "POS_I")
            se_idx = 0
            s_idx = 0

        # 확인자 - 성명
        if all_data_tag[i] == "성명 NNG O" and s_idx == 0:
            s_idx = i + 1
        if all_data_tag[i] == "용도 NNG O" and se_idx == 0:
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

        # 비고 - 회사명
        if all_data_tag[i] == "일 NNBC O" and s_idx == 0:
            s_idx = i + 1
            se_idx = dup_tag4[rp]
            tag = []
            for j in range(s_idx, se_idx):
                tag.append(all_data[j])
                for k in range(0, len(tag)):
                    if k == 0:
                        all_data_tag[j] = (tag[k] + " " + "COM_B")
                    else:
                        all_data_tag[j] = (tag[k] + " " + "COM_I")
            se_idx = 0
            s_idx = 0

        # 비고 - 대표 (직위)
        if all_data_tag[i] == "대표 NNG O" and s_idx == 0 and se_idx == 0:
            s_idx = dup_tag4[rp]
            se_idx = dup_tag4[rp] + 1
            tag = []
            for j in range(s_idx, se_idx):
                tag.append(all_data[j])
                for k in range(0, len(tag)):
                    if k == 0:
                        all_data_tag[j] = (tag[k] + " " + "POS_B")

            s_idx = 0
            se_idx = 0

        # 비고 - 이름
        if all_data_tag[i] == "( SSO O" and se_idx == 0 and s_idx == 0:
            se_idx = dup_tag7[bk]
            s_idx = dup_tag4[rp] + 1
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

    return all_data_tag
