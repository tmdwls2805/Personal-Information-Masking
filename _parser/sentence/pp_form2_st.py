def pp2_st(x):
    import pandas as pd

    data = pd.read_excel(x, header=None)
    data = data.fillna(0)

    final_result = []
    inter_result = []
    first_data = []
    second_data = []

    columns1 = range(8, 24)
    rows = range(0, 10)

    for j in columns1:
        for i in rows:
            if data[i][j] != 0:
                first_data.append(str(data[i][j]))

    second_data.append(data[0][24])
    for i in range(31, 47, 3):
        if data[0][i] != 0:
            second_data.append(data[0][27])
            second_data.append(data[0][i])
        if data[1][i] != 0:
            second_data.append(data[1][27])
            second_data.append(data[1][i])
        if data[4][i] != 0:
            second_data.append(data[4][27])
            second_data.append(data[4][i])
        if data[6][i] != 0:
            second_data.append(data[6][27])
            second_data.append(data[6][i])
        if data[9][i] != 0:
            second_data.append(data[9][27])
            second_data.append(data[9][i])

    inter_result.extend(first_data)
    inter_result.extend(second_data)
    for i in range(0, len(inter_result)):
        inter_result[i] = str(inter_result[i]).replace('\n', '')
        inter_result[i] = inter_result[i].replace('\t', '')
        inter_result[i] = inter_result[i].strip()

    final_result.append(' '.join(inter_result))

    return final_result