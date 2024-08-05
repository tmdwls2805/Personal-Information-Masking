def security3(x):
    from docx import Document
    from _utils.common.Mecab import Mecab

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

        for j in range(len(word_list)):
            word_pos.append(word_list[j] + " " + pos_list[j])
    all_data = list(filter(None, word_pos))
    print(all_data)
    print(len(all_data))

    stsp = []
    for i in range(0, len(all_data)):
        if all_data[i] == ". SF ":
            stsp.append(i)

    restsp = []
    for i in range(0, len(stsp)):
        if i % 2 == 0:
            restsp.append(stsp[i])
#### 수정 필요함 #################################################
    fsss = fstsp_num[0][len(fstsp_num[0]) - 1]

    done = []

    for i in range(0, len(all_data)):
        if i == fsss:
            done.append(all_data[i])
            done.append('')
        elif i == restsp[0]:
            done.append(all_data[i])
            done.append('')
        elif i == restsp[1]:
            done.append(all_data[i])
            done.append('')
        elif i == restsp[2]:
            done.append(all_data[i])
            done.append('')
        elif i == restsp[3]:
            done.append(all_data[i])
            done.append('')
        elif i == restsp[4]:
            done.append(all_data[i])
            done.append('')
        elif i == restsp[5]:
            done.append(all_data[i])
            done.append('')
        elif i == restsp[6]:
            done.append(all_data[i])
            done.append('')
        else:
            done.append(all_data[i])

    return done