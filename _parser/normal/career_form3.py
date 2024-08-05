def career3(x):
    from _utils.common.Mecab import Mecab
    import pandas as pd

    data = pd.read_excel(x, header=None)
    data = data.fillna(0)

    all_data = []
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

    return all_data