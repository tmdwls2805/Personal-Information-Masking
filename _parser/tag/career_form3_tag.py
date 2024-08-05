from _utils.common.Mecab import Mecab
import pandas as pd

def career3_tag(x):

    data = pd.read_excel(x, header=None)
    data = data.fillna(0)

    final_result = []
    first_data = []
    first_process1_data = []
    first_process1_1_data = []
    first_process2_data = []
    first_process2_1_data = []
    first_process3_data = []
    first_process3_1_data = []

    columns = range(8, 35)

    rows = range(0, 10)

    for j in columns:
        for i in rows:
            if data[i][j] != 0:
                first_data.append(data[i][j])

    for i in range(0, 7):
        if len(first_process1_data) % 2 == 0:
            first_process1_data.append(first_data[i])
        else:
            first_process1_data.append(first_data[i].replace(" ", ''))

    for i in range(0, len(first_process1_data)):
        if len(first_process1_1_data) == 2:
            first_process1_1_data.append(first_process1_data[i].replace(" ", ''))
        else:
            first_process1_1_data.append(first_process1_data[i])

    for i in range(7, 30):
        if len(first_process2_data) % 2 == 0:
            first_process2_data.append(first_data[i])
        else:
            first_process2_data.append(first_data[i].replace(" ", ''))

    for i in range(0, len(first_process2_data)):
        if len(first_process2_1_data) == 6:
            first_process2_1_data.append(first_process2_data[i].replace(" ", ''))
        elif len(first_process2_1_data) == 14:
            first_process2_1_data.append(first_process2_data[i].replace(" ", ''))
        else:
            first_process2_1_data.append(first_process2_data[i])

    for i in range(30, 39):
        if len(first_process3_data) % 2 == 0:
            first_process3_data.append(first_data[i])
        else:
            first_process3_data.append(first_data[i].replace(" ", ''))

    for i in range(0, len(first_process3_data)):
        if len(first_process3_1_data) == 2:
            first_process3_1_data.append(first_process3_data[i].replace(" ", ''))
        elif len(first_process3_1_data) == 4:
            first_process3_1_data.append(first_process3_data[i].replace(" ", ''))
        else:
            first_process3_1_data.append(first_process3_data[i])

    final_result.extend(first_process1_1_data)
    final_result.extend(first_process2_1_data)
    final_result.extend(first_process3_1_data)
    final_result.append(first_data[39])

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
            if word_list[i] == '주민':
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
        if all_data_tag[n] == "경력 NNG O":
            dup_tag.append(n)

        if all_data_tag[n] == "담당 NNG O":
            dup_tag2.append(n)

        if all_data_tag[n] == "( SSO O":
            dup_tag3.append(n)

        if all_data_tag[n] == "업종 NNG O":
            dup_tag4.append(n)

        if all_data_tag[n] == "부서 NNG O":
            dup_tag5.append(n)

        if all_data_tag[n] == "일 NNBC O":
            dup_tag6.append(n)

        if all_data_tag[n] == "대표 NNG O":
            dup_tag7.append(n)

    for i in range(len(all_data_tag)):

        text = all_data_tag[i]

        # 인적사항 - 이름
        if all_data_tag[i] == "성명 NNG O" and s_idx == 0:
            s_idx = i + 1
        if all_data_tag[i] == "주민 NNG O" and se_idx == 0:
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

        # 인적사항 - 주소
        if all_data_tag[i] == "주소 NNG O" and s_idx == 0 and se_idx == 0:
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

        # 경력사항 - 대표자
        if all_data_tag[i] == "대표자 NNG O" and s_idx == 0 and se_idx == 0:
            s_idx = i + 1
            se_idx = dup_tag4[0]
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

        # 경력사항 - 소재지(주소)
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
        if all_data_tag[i] == "직위 NNG O" and se_idx == 0:
            s_idx = dup_tag5[0] + 1
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
        if all_data_tag[i] == "직위 NNG O" and s_idx == 0:
            tag = []
            s_idx = i + 1
            if len(dup_tag2) == 2:
                se_idx = dup_tag2[0]
                for j in range(s_idx, se_idx):
                    tag.append(all_data[j])
                    for k in range(0, len(tag)):
                        if k == 0:
                            all_data_tag[j] = (tag[k] + " " + "POS_B")
                        else:
                            all_data_tag[j] = (tag[k] + " " + "POS_I")
            elif len(dup_tag2) == 4:
                se_idx = dup_tag2[1]
                for j in range(s_idx, se_idx):
                    tag.append(all_data[j])
                    for k in range(0, len(tag)):
                        if k == 0:
                            all_data_tag[j] = (tag[k] + " " + "POS_B")
                        else:
                            all_data_tag[j] = (tag[k] + " " + "POS_I")
            elif len(dup_tag2) == 3:
                if s_idx == dup_tag2[0]:
                    se_idx = dup_tag2[1]
                    for j in range(s_idx, se_idx):
                        tag.append(all_data[j])
                        for k in range(0, len(tag)):
                            if k == 0:
                                all_data_tag[j] = (tag[k] + " " + "POS_B")
                            else:
                                all_data_tag[j] = (tag[k] + " " + "POS_I")
                else:
                    se_idx = dup_tag2[0]
                    for j in range(s_idx, se_idx):
                        tag.append(all_data[j])
                        for k in range(0, len(tag)):
                            if k == 0:
                                all_data_tag[j] = (tag[k] + " " + "POS_B")
                            else:
                                all_data_tag[j] = (tag[k] + " " + "POS_I")
            s_idx = 0
            se_idx = 0

        # 발급처 - 담당부서
        if all_data_tag[i] == "담당자 NNG O" and se_idx == 0 and s_idx == 0:
            s_idx = dup_tag5[1] + 1
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

        # 발급처 - 담당자
        if all_data_tag[i] == "담당자 NNG O" and s_idx == 0:
            s_idx = i + 1
        if all_data_tag[i] == "직책 NNG O" and se_idx == 0:
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

        # 발급처 - 직책
        if all_data_tag[i] == "직책 NNG O" and s_idx == 0:
            s_idx = i + 1
        if all_data_tag[i] == "연락처 NNG O" and se_idx == 0:
            se_idx = i
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
        if all_data_tag[i] == "대표 NNG O" and s_idx == 0 and se_idx == 0:
            s_idx = dup_tag6[len(dup_tag6) - 1] + 1
            se_idx = dup_tag7[len(dup_tag7) - 1]
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

        # 비고 - 대표(직위)
        if all_data_tag[i] == "대표 NNG O" and s_idx == 0 and se_idx == 0:
            s_idx = dup_tag7[len(dup_tag7) - 1]
            se_idx = dup_tag7[len(dup_tag7) - 1] + 1
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

        # 비고 - 이름
        if all_data_tag[i] == "( SSO O" and s_idx == 0 and se_idx == 0:
            s_idx = dup_tag7[len(dup_tag7) - 1] + 1
            se_idx = dup_tag3[len(dup_tag3) - 1]
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

    return all_data_tag
