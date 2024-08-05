def career1(x):
    import pandas as pd
    from _utils.common.Mecab import Mecab

    data = pd.read_excel(x, header=None)
    data = data.fillna(0)

    all_data = []
    first_data = []
    second_data = []
    third_data = []
    final_result = []

    columns1 = range(8, 15)
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

    return all_data