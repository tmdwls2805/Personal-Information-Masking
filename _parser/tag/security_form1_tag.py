from docx import Document
from _utils.common.Mecab import Mecab
from _parser.utils import ngram
from _parser.utils import sel_sort

def security1_tag(x):

    doc = Document(x)
    result = [p.text for p in doc.paragraphs]
    final_result = [v for v in result if v]

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
        b_idx = []
        e_idx = []
        for j in range(len(word_list)):
            if word_list[j] == "은":
                b_idx.append(j)
                e_idx.append(word_list[j])
        for j in range(len(b_idx)):
            del (word_list[b_idx[j]])
            del (pos_list[b_idx[j]])
            word_list.insert(b_idx[j], e_idx[j])
            pos_list.insert(b_idx[j], "JX")

        b_idx = []
        e_idx = []
        for j in range(len(word_list)):
            if word_list[j] == "에":
                b_idx.append(j)
                e_idx.append(word_list[j])
        for j in range(len(b_idx)):
            del (word_list[b_idx[j]])
            del (pos_list[b_idx[j]])
            word_list.insert(b_idx[j], e_idx[j])
            pos_list.insert(b_idx[j], "JKB")

        b_idx = []
        e_idx = []
        for j in range(len(word_list)):
            if word_list[j] == "에서":
                b_idx.append(j)
                e_idx.append(word_list[j])
        for j in range(len(b_idx)):
            del (word_list[b_idx[j]])
            del (pos_list[b_idx[j]])
            word_list.insert(b_idx[j], e_idx[j])
            pos_list.insert(b_idx[j], "JKB")

        b_idx = []
        e_idx = []
        for j in range(len(word_list)):
            if word_list[j] == ".":
                b_idx.append(j)
                e_idx.append(word_list[j])
        for j in range(len(b_idx)):
            del (word_list[b_idx[j]])
            del (pos_list[b_idx[j]])
            word_list.insert(b_idx[j], e_idx[j])
            pos_list.insert(b_idx[j], "SF")


        for j in range(len(word_list)):
            word_pos.append(word_list[j] + " " + pos_list[j])
    all_data = list(filter(None, word_pos))
    all_data_none = list(filter(None, word_list))

    all_data_none_num = []
    for i in range(len(all_data_none)):
        all_data_none_num.append(i)

    all_data_tag = []
    dup_tag = []
    dup_tag2 = []
    dup_tag3 = []
    dup_tag4 = []
    dup_tag5 = []
    dup_tag6 = []
    dup_tag7 = []
    dup_tag8 = []
    tag = []
    none_tag = []
    snone_tag = []
    tag_ex = []
    fresult = []
    fresult_num = []
    sresult_num = []
    jresult_num = []
    s_idx = 0
    se_idx = 0
    for i in range(len(all_data)):
        all_data_tag.append(all_data[i] + " " + "O")

    for i in range(len(all_data_tag)):
        if all_data_tag[i] == '은 JX O':
            dup_tag.append(i)
        if all_data_tag[i] == "는 ETM O":
            dup_tag2.append(i)
        if all_data_tag[i] == "에서 JKB O":
            dup_tag3.append(i)
        if all_data_tag[i] == "에 JKB O":
            dup_tag4.append(i)
        if all_data_tag[i] == ") SSC O":
            dup_tag5.append(i)
        if all_data_tag[i] == ": SC O":
            dup_tag6.append(i)
        if all_data_tag[i] == "서약 NNG O":
            dup_tag7.append(i)
        if all_data_tag[i] == "에게 JKB O":
            dup_tag8.append(i)
    for i in range(len(all_data_tag)):
        # 이름 - 2개
        if all_data_tag[i] == "본인 NNG O" and s_idx == 0 and se_idx == 0:
            s_idx = i + 1
            se_idx = dup_tag[0]
            for j in range(s_idx, se_idx):
                none_tag.append(all_data_none[j])
                tag.append(all_data[j])

    jnone_tag = ''.join(none_tag)  # ngram 시킨 값과의 비교 대상
    snone_tag = list(jnone_tag)  # ngram을 몇 번 반복할지에 대한 여부 (글자 수 만큼)
    for k in range(1, len(snone_tag)+1):

        nresult = ngram(all_data_none, k)   # ngram(1 ~ 찾을 단어의 글자 수)
        for n in range(len(nresult)):
            jresult = ''.join(nresult[n])
            fresult.append(jresult)

        nresult_num = ngram(all_data_none_num, k)   # ngram(찾을 단어의 글자의 위치)
        jresult_num.append(nresult_num)

    for i in range(0, len(jresult_num)):
        for j in range(0, len(jresult_num[i])):
            fresult_num.append(jresult_num[i][j])

    for i in range(0, len(fresult)):
        if fresult[i] == jnone_tag:
            sresult_num.append(fresult_num[i])

    for i in range(0, len(sresult_num)):
        for j in range(0, len(sresult_num[i])):
            if j == 0:
                all_data_tag[sresult_num[i][j]] = (all_data[sresult_num[i][j]] + " " + "PER_B")
            else:
                all_data_tag[sresult_num[i][j]] = (all_data[sresult_num[i][j]] + " " + "PER_I")
    tag = []
    none_tag = []
    snone_tag = []
    tag_ex = []
    fresult = []
    fresult_num = []
    sresult_num = []
    jresult_num = []
    s_idx = 0
    se_idx = 0

    for i in range(len(all_data_tag)):
        # 회사명 - 10개
        if all_data_tag[i] == "는 ETM O" and s_idx == 0 and se_idx == 0:
            s_idx = dup_tag2[0] + 1
            se_idx = dup_tag3[0] #에서
            for j in range(s_idx, se_idx):
                none_tag.append(all_data_none[j])
                tag.append(all_data[j])


    jnone_tag = ''.join(none_tag)  # ngram 시킨 값과의 비교 대상
    snone_tag = list(jnone_tag)  # ngram을 몇 번 반복할지에 대한 여부 (글자 수 만큼)

    for k in range(1, len(snone_tag)+1):

        nresult = ngram(all_data_none, k)   # ngram(1 ~ 찾을 단어의 글자 수)
        for n in range(len(nresult)):
            jresult = ''.join(nresult[n])
            fresult.append(jresult)

        nresult_num = ngram(all_data_none_num, k)   # ngram(찾을 단어의 글자의 위치)
        jresult_num.append(nresult_num)

    for i in range(0, len(jresult_num)):
        for j in range(0, len(jresult_num[i])):
            fresult_num.append(jresult_num[i][j])

    for i in range(0, len(fresult)):
        if fresult[i] == jnone_tag:
            sresult_num.append(fresult_num[i])

    for i in range(0, len(sresult_num)):
        for j in range(0, len(sresult_num[i])):
            if j == 0:
                all_data_tag[sresult_num[i][j]] = (all_data[sresult_num[i][j]] + " " + "COM_B")
            else:
                all_data_tag[sresult_num[i][j]] = (all_data[sresult_num[i][j]] + " " + "COM_I")
    tag = []
    none_tag = []
    snone_tag = []
    tag_ex = []
    fresult = []
    fresult_num = []
    jresult_num = []
    s_idx = 0
    se_idx = 0

    sresult_num = sel_sort(sresult_num)     # 선택 정렬로 순서대로 정렬함
    faff = sresult_num[4][len(sresult_num[4]) - 1]
    for i in range(len(all_data_tag)):
        # 소속1 - 회사명 5번 째 뒤
        if all_data_tag[i] == all_data_tag[faff-1] and s_idx == 0 and se_idx == 0:
            s_idx = faff + 1
            se_idx = dup_tag8[0]
            for j in range(s_idx, se_idx):
                none_tag.append(all_data_none[j])
                tag.append(all_data[j])
                for k in range(0, len(tag)):
                    if k == 0:
                        all_data_tag[j] = (tag[k] + " " + "AFF_B")
                    else:
                        all_data_tag[j] = (tag[k] + " " + "AFF_I")
    s_idx = 0
    se_idx = 0
    faff = 0
    tag = []
    faff = sresult_num[6][len(sresult_num[6]) - 1]
    for i in range(len(all_data_tag)):
        # 소속1 - 회사명 5번 째 뒤
        if all_data_tag[i] == all_data_tag[faff-1] and s_idx == 0 and se_idx == 0:
            s_idx = faff + 1
            se_idx = dup_tag8[len(dup_tag8)-1]
            for j in range(s_idx, se_idx):
                none_tag.append(all_data_none[j])
                tag.append(all_data[j])
                for k in range(0, len(tag)):
                    if k == 0:
                        all_data_tag[j] = (tag[k] + " " + "AFF_B")
                    else:
                        all_data_tag[j] = (tag[k] + " " + "AFF_I")

    sresult_num = []
    tag = []
    s_idx = 0
    se_idx = 0

    for i in range(len(all_data_tag)):
        # 주소1
        if all_data_tag[i] == "에 JKB O" and s_idx == 0 and se_idx == 0:
            s_idx = dup_tag[0] + 1
            se_idx = dup_tag4[0]
            for j in range(s_idx, se_idx):
                tag.append(all_data[j])
                for k in range(0, len(tag)):
                    if k == 0:
                        all_data_tag[j] = (tag[k] + " " + "LOC_B")
                    else:
                        all_data_tag[j] = (tag[k] + " " + "LOC_I")
        s_idx = 0
        se_idx = 0
        tag = []

        # 주소2
        if all_data_tag[i] == ": SC O" and s_idx == 0 and se_idx == 0:
            s_idx = dup_tag6[0] + 1
            se_idx = dup_tag7[2]
            for j in range(s_idx, se_idx):
                tag.append(all_data[j])
                for k in range(0, len(tag)):
                    if k == 0:
                        all_data_tag[j] = (tag[k] + " " + "LOC_B")
                    else:
                        all_data_tag[j] = (tag[k] + " " + "LOC_I")

        s_idx = 0
        se_idx = 0
        tag = []

    stsp = []
    for i in range(0, len(all_data_tag)):
        if all_data_tag[i] == ". SF O":
            stsp.append(i)

    restsp = []
    for i in range(0, len(stsp)):
        if i % 2 == 0:
            restsp.append(stsp[i])

    done = []

    for i in range(0, len(all_data_tag)):
        if i == restsp[0]:
            done.append(all_data_tag[i])
            done.append('')
        elif i == restsp[1]:
            done.append(all_data_tag[i])
            done.append('')
        elif i == restsp[2]:
            done.append(all_data_tag[i])
            done.append('')
        elif i == restsp[3]:
            done.append(all_data_tag[i])
            done.append('')
        elif i == restsp[4]:
            done.append(all_data_tag[i])
            done.append('')
        elif i == restsp[5]:
            done.append(all_data_tag[i])
            done.append('')
        else:
            done.append(all_data_tag[i])

    return done