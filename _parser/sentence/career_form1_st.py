def career1_st(x):
    import pandas as pd

    data = pd.read_excel(x, header=None)
    data = data.fillna(0)

    first_data = []
    second_data = []
    third_data = []
    inter_result = []
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
        inter_result.extend(first_data)
        inter_result.extend(third_data)

    elif len(second_data) != 0:
        inter_result.extend(first_data)
        inter_result.extend(second_data)
        inter_result.extend(third_data)

    for i in range(0, len(inter_result)):
        inter_result[i] = inter_result[i].replace('\n', '')
        inter_result[i] = inter_result[i].replace('\t', '')
        inter_result[i] = inter_result[i].strip()


    final_result.append(' '.join(inter_result))

    return final_result