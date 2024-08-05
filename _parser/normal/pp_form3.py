def pp3(x):
    import pandas as pd
    from _utils.common.Mecab import Mecab

    data = pd.read_excel(x, header=None)
    data = data.fillna(0)

    all_data = []
    first_data = []
    second_data = []
    final_result = []

    columns1 = range(8, 17)
    rows = range(0, 10)

    for j in columns1:
        for i in rows:
            if data[i][j] != 0:
                first_data.append(str(data[i][j]))

    second_data.append(data[0][19])
    for i in range(23, 42, 2):
        if data[0][i] != 0:
            second_data.append(data[0][21])
            second_data.append(data[0][i])
        if data[3][i] != 0:
            second_data.append(data[3][21])
            second_data.append(data[3][i])
        if data[5][i] != 0:
            second_data.append(data[5][21])
            second_data.append(data[5][i])
        if data[7][i] != 0:
            second_data.append(data[7][21])
            second_data.append(data[7][i])
        if data[8][i] != 0:
            second_data.append(data[8][21])
            second_data.append(data[8][i])
        if data[9][i] != 0:
            second_data.append(data[9][21])
            second_data.append(data[9][i])

    final_result.extend(first_data)
    final_result.extend(second_data)

    for i in range(0, len(final_result)):
        final_result[i] = str(final_result[i])

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
            if word_list[i] == "소속":
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