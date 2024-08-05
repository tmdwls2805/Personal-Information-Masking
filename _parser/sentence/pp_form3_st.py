def pp3_st(x):
    import pandas as pd

    data = pd.read_excel(x, header=None)
    data = data.fillna(0)

    first_data = []
    second_data = []
    inter_result = []
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

    inter_result.extend(first_data)
    inter_result.extend(second_data)

    for i in range(0, len(inter_result)):
        inter_result[i] = str(inter_result[i]).replace('\n', '')
        inter_result[i] = inter_result[i].replace('\t', '')
        inter_result[i] = inter_result[i].strip()

    final_result.append(' '.join(inter_result))

    return final_result